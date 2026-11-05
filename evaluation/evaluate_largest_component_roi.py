"""Controlled, NON-DESTRUCTIVE 5-fold evaluation of largest-component ROI cleanup.

Run from the HAM10000 project root:
  python -m evaluation.evaluate_largest_component_roi --fold all

Does not train or overwrite existing crops/checkpoints/reports. It recomputes U-Net
masks using the original exporter preprocessing, retains only the largest
8-connected region, and evaluates altered ROI crops with the SAME fold-specific
reference-ROI-trained ConvNeXt checkpoint. Original predicted-mask predictions
are loaded from saved per-image comparison CSVs; altered cases replay those
original crops and assert exact agreement before computing cleaned predictions.

This comparison is EXPLORATORY: the hypothesis was developed after inspecting
Fold 5. Do not present the all-fold result as an unbiased independent test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from core.canonical_cv import get_fold_metadata
from core.segmentation_unet import SmallUNet
from preprocessing.roi_crop import crop_lesion_with_margin

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
RUN = "convnext_tiny_natural_ce_roi_margin15_seed123"


def mask_after_largest_component(binary: np.ndarray) -> tuple[np.ndarray, int, float]:
    labels, number = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if number == 0:
        return binary.copy(), 0, 0.0
    counts = np.bincount(labels.ravel(), minlength=number + 1)
    counts[0] = 0
    result = labels == int(counts.argmax())
    fraction_removed = float((binary.sum() - result.sum()) / binary.sum())
    return result, number, fraction_removed


def jpeg_roundtrip(image: Image.Image) -> Image.Image:
    """Mirror original exporter: RGB.crop.resize((224,224)).save(.jpg)."""
    buf = BytesIO()
    image.resize((224, 224)).save(buf, format="JPEG")
    buf.seek(0)
    with Image.open(buf) as saved:
        return saved.convert("RGB")


def model_predictions(model, images, device, batch_size):
    if not images:
        return []
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            data = torch.stack(images[start:start + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(data)
            result.extend(logits.argmax(dim=1).cpu().tolist())
    return result


def load_comparison(fold, val):
    path = Path(f"reports/segmentation/small_unet_fold{fold}_seed123/") / (
        f"classifier_predicted_vs_reference_fold{fold}.csv"
    )
    original = pd.read_csv(path, dtype={"image_id": "str"})
    needed = {"image_id", "dx", "true_label", "reference_prediction", "predicted_mask_prediction"}
    if not needed.issubset(original):
        raise ValueError(f"Columns missing in {path}: {sorted(needed - set(original))}")
    if original.image_id.duplicated().any():
        raise ValueError(f"Duplicate image IDs in {path}")
    if set(original.image_id) != set(val.image_id):
        raise ValueError(f"Image IDs differ from canonical validation in Fold {fold}")
    original = original.set_index("image_id")
    for row in val.itertuples(index=False):
        entry = original.loc[row.image_id]
        if entry.dx != row.dx or int(entry.true_label) != CLASS_TO_IDX[row.dx]:
            raise ValueError(f"Label mismatch for {row.image_id}")
    return original


def evaluate_fold(fold, batch_size, device):
    print(f"\n=== Fold {fold} / largest-component ROI ===", flush=True)
    report_dir = Path(f"reports/segmentation/small_unet_fold{fold}_seed123")
    csv_out = report_dir / f"largest_component_roi_comparison_fold{fold}.csv"
    json_out = report_dir / f"largest_component_roi_comparison_fold{fold}_summary.json"
    if csv_out.exists() or json_out.exists():
        raise FileExistsError(f"Refusing to overwrite {csv_out} or {json_out}")

    _, val = get_fold_metadata(fold)
    val = val.reset_index(drop=True)
    previous = load_comparison(fold, val)

    seg_path = Path(f"checkpoints/segmentation/small_unet_fold{fold}_seed123_best.pt")
    cls_path = Path(f"checkpoints/{RUN}_fold{fold}_best.pt")
    if not seg_path.is_file() or not cls_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {seg_path} or {cls_path}")

    segmenter = SmallUNet().to(device).eval()
    segmenter.load_state_dict(torch.load(seg_path, map_location=device, weights_only=True)["model_state_dict"])
    classifier = convnext_tiny(weights=None)
    classifier.classifier[2] = nn.Linear(classifier.classifier[2].in_features, 7)
    classifier.load_state_dict(torch.load(cls_path, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
    classifier = classifier.to(device).eval()
    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()

    rows = []
    pending_old, pending_new, pending_pos = [], [], []

    def flush_pending():
        if not pending_pos:
            return
        orig_preds = model_predictions(classifier, pending_old, device, batch_size)
        new_preds = model_predictions(classifier, pending_new, device, batch_size)
        for pos, p_before, p_after in zip(pending_pos, orig_preds, new_preds):
            if p_before != rows[pos]["original_predicted_roi_prediction"]:
                raise AssertionError(
                    f"Saved original prediction mismatch: fold={fold}, "
                    f"id={rows[pos]['image_id']}, saved={rows[pos]['original_predicted_roi_prediction']}, "
                    f"replayed={p_before}. Stop to inspect transforms/checkpoint."
                )
            rows[pos]["largest_component_roi_prediction"] = int(p_after)
        pending_pos.clear(); pending_old.clear(); pending_new.clear()

    with torch.inference_mode():
        for i, row in enumerate(val.itertuples(index=False), 1):
            original_entry = previous.loc[row.image_id]
            original_prediction = int(original_entry.predicted_mask_prediction)
            image_path = Path("data/images") / f"{row.image_id}.jpg"
            saved_crop = (Path(f"data/processed_roi_predicted_fold{fold}_v1")
                          / "val" / row.dx / f"{row.image_id}_224x224.jpg")
            if not saved_crop.is_file():
                raise FileNotFoundError(saved_crop)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            resized = image.resize((256, 192), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(device)
            predicted = (torch.sigmoid(segmenter(tensor))[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
            mask = Image.fromarray(predicted * 255).resize(image.size, Image.Resampling.NEAREST)
            binary = np.asarray(mask, dtype=np.uint8) > 127
            cleaned_binary, n_components, fraction_removed = mask_after_largest_component(binary)
            cleaned_mask = Image.fromarray(cleaned_binary.astype(np.uint8) * 255)
            old_crop, old_status = crop_lesion_with_margin(image, mask)
            new_crop, new_status = crop_lesion_with_margin(image, cleaned_mask)
            changed = not np.array_equal(np.asarray(old_crop), np.asarray(new_crop))

            y = CLASS_TO_IDX[row.dx]
            record = {
                "fold": fold, "image_id": row.image_id, "dx": row.dx,
                "true_label": y,
                "reference_roi_prediction": int(original_entry.reference_prediction),
                "original_predicted_roi_prediction": original_prediction,
                "largest_component_roi_prediction": original_prediction,
                "components": int(n_components),
                "removed_foreground_fraction": fraction_removed,
                "original_status": old_status, "largest_only_status": new_status,
                "crop_changed_by_filter": bool(changed),
                "status_changed_by_filter": old_status != new_status,
                "original_crop_fraction": old_crop.width * old_crop.height / (image.width * image.height),
                "largest_crop_fraction": new_crop.width * new_crop.height / (image.width * image.height),
            }
            rows.append(record)
            if changed:
                with Image.open(saved_crop) as stored:
                    old_tensor = transform(stored.convert("RGB"))
                new_tensor = transform(jpeg_roundtrip(new_crop))
                pending_old.append(old_tensor)
                pending_new.append(new_tensor)
                pending_pos.append(len(rows) - 1)
                if len(pending_pos) >= batch_size:
                    flush_pending()
            if i % 250 == 0 or i == len(val):
                print(f"Fold {fold}: evaluated {i}/{len(val)}", flush=True)
        flush_pending()

    output = pd.DataFrame(rows)
    true = output.true_label.to_numpy(dtype=int)
    reference = output.reference_roi_prediction.to_numpy(dtype=int)
    old = output.original_predicted_roi_prediction.to_numpy(dtype=int)
    new = output.largest_component_roi_prediction.to_numpy(dtype=int)
    # These are changes compared with the *original predicted-mask* ROI pipeline.
    output["outcome_vs_original_predicted_roi"] = np.select(
        [(old != true) & (new == true), (old == true) & (new != true),
         (old != true) & (new != true) & (old != new)],
        ["corrected", "regression", "wrong_to_other_wrong"], default="unchanged")
    output["prediction_changed_by_filter"] = old != new
    macro = lambda predictions: float(f1_score(true, predictions, labels=list(range(7)),
                                              average="macro", zero_division=0))
    per_class = {}
    for i, name in enumerate(CLASSES):
        per_class[name] = {
            "original_predicted_roi_f1": float(f1_score(true, old, labels=[i], average=None, zero_division=0)[0]),
            "largest_component_roi_f1": float(f1_score(true, new, labels=[i], average=None, zero_division=0)[0]),
        }
        per_class[name]["delta"] = (per_class[name]["largest_component_roi_f1"]
                                    - per_class[name]["original_predicted_roi_f1"])
    summary = {
        "fold": fold, "n_images": len(output),
        "reference_roi_macro_f1": macro(reference),
        "original_predicted_roi_macro_f1": macro(old),
        "largest_component_roi_macro_f1": macro(new),
        "delta_vs_original_predicted_roi": macro(new) - macro(old),
        "original_accuracy": float(accuracy_score(true, old)),
        "largest_accuracy": float(accuracy_score(true, new)),
        "n_multicomponent": int((output.components > 1).sum()),
        "n_changed_crops": int(output.crop_changed_by_filter.sum()),
        "n_changed_statuses": int(output.status_changed_by_filter.sum()),
        "n_changed_class_predictions": int(output.prediction_changed_by_filter.sum()),
        "outcomes_vs_original_predicted_roi": dict(Counter(output.outcome_vs_original_predicted_roi)),
        "original_roi_status_counts": dict(Counter(output.original_status)),
        "largest_component_status_counts": dict(Counter(output.largest_only_status)),
        "per_class": per_class,
        "note": "Exploratory post-hoc controlled evaluation; not independent external validation.",
    }
    # Check Fold 5 component-audit agreement if the audit CSV is available.
    audit_file = report_dir / f"mask_component_audit_fold{fold}.csv"
    if audit_file.exists():
        audit = pd.read_csv(audit_file, dtype={"image_id": str})
        if (len(audit) != len(output) or set(audit.image_id) != set(output.image_id)):
            raise AssertionError("Existing component audit image IDs disagree")
        audit = audit.set_index("image_id").loc[output.image_id]
        for old_col, new_col in [
            ("crop_changed_by_largest_only", "crop_changed_by_filter"),
            ("status_changed_by_largest_only", "status_changed_by_filter"),
            ("original_status", "original_status"),
            ("largest_only_status", "largest_only_status"),
        ]:
            if not np.array_equal(audit[old_col].to_numpy(), output[new_col].to_numpy()):
                raise AssertionError(f"Existing audit mismatch: {old_col}")

    report_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2), flush=True)
    print(f"Saved: {csv_out} and {json_out}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["1", "2", "3", "4", "5", "all"], default="5")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = list(range(1, 6)) if args.fold == "all" else [int(args.fold)]
    for f in folds:
        parent = Path(f"reports/segmentation/small_unet_fold{f}_seed123")
        for path in [parent / f"largest_component_roi_comparison_fold{f}.csv",
                     parent / f"largest_component_roi_comparison_fold{f}_summary.json"]:
            if path.exists():
                raise FileExistsError(f"Output already exists, refusing to overwrite: {path}")
    results = [evaluate_fold(f, args.batch_size, device) for f in folds]
    if len(results) == 5:
        out = Path("reports/segmentation/largest_component_roi_5fold_summary.json")
        if out.exists():
            raise FileExistsError(f"Output already exists: {out}")
        vals = np.array([r["delta_vs_original_predicted_roi"] for r in results])
        whole = {
            "folds": results,
            "mean_paired_macro_f1_delta": float(vals.mean()),
            "sample_sd_paired_macro_f1_delta": float(vals.std(ddof=1)),
            "note": "Hypothesis designed after Fold 5 inspection; exploratory, not an unbiased test.",
        }
        out.write_text(json.dumps(whole, indent=2), encoding="utf-8")
        print("\nFive-fold mean paired delta:", round(float(vals.mean()), 6),
              "+/-", round(float(vals.std(ddof=1)), 6))
        print("Saved:", out)


if __name__ == "__main__":
    main()
