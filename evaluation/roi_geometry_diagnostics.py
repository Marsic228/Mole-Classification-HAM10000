"""Read-only audit of reference-mask vs predicted-mask ROI geometry.

Run from Mole Classification repository root, e.g.:
    python -m evaluation.roi_geometry_diagnostics --fold 5

This script does NOT regenerate or overwrite ROI JPEGs and does NOT modify checkpoints.
The logic in geometry() mirrors preprocessing.roi_crop.crop_lesion_with_margin;
its status and crop dimensions are checked against that original function.
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from core.canonical_cv import get_fold_metadata
from core.segmentation_unet import SmallUNet
from preprocessing.roi_crop import crop_lesion_with_margin

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def geometry(mask: Image.Image, margin=0.15, min_fraction=0.35, max_area=0.60):
    """Record raw bbox, candidate crop, final used crop and exact fallback reason."""
    width, height = mask.size
    binary = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(binary)
    count = int(binary.sum())
    area = count / (width * height)
    fields = {
        "status": "fallback_empty", "foreground_pixels": count,
        "foreground_fraction": area, "bbox_left": None, "bbox_top": None,
        "bbox_right": None, "bbox_bottom": None, "bbox_fraction": None,
        "touches_edge": False, "candidate_width": None,
        "candidate_height": None, "candidate_fraction": None,
        "crop_left": 0, "crop_top": 0,
        "crop_right": width, "crop_bottom": height,
        "crop_width": width, "crop_height": height,
        "crop_fraction": 1.0,
    }
    if count == 0:
        return fields

    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    edge = x0 == 0 or y0 == 0 or x1 == width or y1 == height
    crop_w = max((x1 - x0) * (1 + 2 * margin), width * min_fraction)
    crop_h = max((y1 - y0) * (1 + 2 * margin), height * min_fraction)
    aspect = width / height
    crop_w = max(crop_w, crop_h * aspect)
    crop_h = crop_w / aspect
    fields.update({
        "bbox_left": x0, "bbox_top": y0,
        "bbox_right": x1, "bbox_bottom": y1,
        "bbox_fraction": ((x1 - x0) * (y1 - y0)) / (width * height),
        "touches_edge": edge,
        "candidate_width": crop_w, "candidate_height": crop_h,
        "candidate_fraction": (crop_w * crop_h) / (width * height),
    })
    if area > max_area:
        fields["status"] = "fallback_large_area"
    elif edge:
        fields["status"] = "fallback_edge"
    elif crop_w > width or crop_h > height:
        fields["status"] = "fallback_large_crop"
    else:
        crop_w, crop_h = math.ceil(crop_w), math.ceil(crop_h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left = max(0, min(round(cx - crop_w / 2), width - crop_w))
        top = max(0, min(round(cy - crop_h / 2), height - crop_h))
        fields.update({
            "status": "roi", "crop_left": left, "crop_top": top,
            "crop_right": left + crop_w, "crop_bottom": top + crop_h,
            "crop_width": crop_w, "crop_height": crop_h,
            "crop_fraction": (crop_w * crop_h) / (width * height),
        })
    return fields


def add_prefix(out, prefix, values):
    out.update({f"{prefix}_{key}": value for key, value in values.items()})


def mask_overlap(reference, predicted):
    ref = np.asarray(reference.convert("L")) > 127
    pred = np.asarray(predicted.convert("L")) > 127
    overlap = int(np.logical_and(ref, pred).sum())
    denom = int(ref.sum()) + int(pred.sum())
    union = int(np.logical_or(ref, pred).sum())
    return (2 * overlap / denom if denom else 1.0,
            overlap / union if union else 1.0, ref)


def pixel_retention(reference_binary, predicted_geometry):
    denom = int(reference_binary.sum())
    if not denom:
        return None
    g = predicted_geometry
    return float(reference_binary[
        g["crop_top"]:g["crop_bottom"],
        g["crop_left"]:g["crop_right"],
    ].sum() / denom)


def compare_outcome(row, label):
    ref, pred = int(row.reference_prediction), int(row.predicted_mask_prediction)
    if ref == pred:
        return "unchanged"
    if ref == label:
        return "regression"
    if pred == label:
        return "corrected"
    return "wrong_to_other_wrong"


def load_comparison(path, fold, val):
    if not path.is_file():
        raise FileNotFoundError(f"Missing classification comparison: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    required = {"image_id", "dx", "true_label", "reference_prediction", "predicted_mask_prediction"}
    if not required.issubset(df.columns):
        raise ValueError(f"Comparison CSV lacks {required - set(df.columns)}")
    if df.image_id.isna().any() or not df.image_id.is_unique:
        raise ValueError("Comparison CSV has missing/duplicate image_id")
    if set(df.image_id) != set(val.image_id):
        raise ValueError("Comparison IDs don't match the canonical held-out fold")
    labels = {key: i for i, key in enumerate(CLASSES)}
    truth = val.set_index("image_id")["dx"].to_dict()
    if any(row.dx != truth[row.image_id] or int(row.true_label) != labels[row.dx]
           for row in df.itertuples(index=False)):
        raise ValueError("Comparison truth/labels differ from frozen fold")
    df["outcome"] = [compare_outcome(row, labels[row.dx]) for row in df.itertuples(index=False)]
    df["fold"] = fold
    return df.set_index("image_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--images-root", type=Path, default=Path("data/images"))
    parser.add_argument("--mapping", type=Path, default=Path("Masks_to_IsicId_mapping.csv"))
    parser.add_argument("--masks", type=Path, default=Path("Masks.zip"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--comparison", type=Path, default=None)
    args = parser.parse_args()

    fold = args.fold
    output_dir = args.output_dir or Path(f"reports/segmentation/small_unet_fold{fold}_seed123")
    output_csv = output_dir / f"roi_geometry_diagnostic_fold{fold}.csv"
    output_json = output_dir / f"roi_geometry_diagnostic_fold{fold}_summary.json"
    for output in (output_csv, output_json):
        if output.exists():
            raise FileExistsError(f"Diagnostic already exists; refusing overwrite: {output}")

    _, val = get_fold_metadata(fold)
    if not val.image_id.is_unique:
        raise ValueError("Fold validation image IDs are not unique")
    comparison_path = args.comparison or output_dir / f"classifier_predicted_vs_reference_fold{fold}.csv"
    comparison = load_comparison(comparison_path, fold, val)

    mapping = pd.read_csv(args.mapping, dtype={"ISIC_ID": str})
    if not {"ISIC_ID", "Filename"}.issubset(mapping.columns):
        raise ValueError("Mask mapping requires ISIC_ID and Filename")
    if mapping.ISIC_ID.isna().any() or not mapping.ISIC_ID.is_unique:
        raise ValueError("Mask mapping has missing/duplicate ISIC_ID")
    filenames = mapping.set_index("ISIC_ID")["Filename"].to_dict()
    if set(val.image_id) - set(filenames):
        raise ValueError("Reference mask mapping lacks held-out image IDs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = Path(f"checkpoints/segmentation/small_unet_fold{fold}_seed123_best.pt")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = []
    with ZipFile(args.masks) as archive, torch.inference_mode():
        # Match masks by basename, allowing either Masks/name.png or name.png.
        archive_masks = {}
        for member in archive.namelist():
            if member.lower().endswith(".png"):
                name = Path(member).name
                if name in archive_masks:
                    raise ValueError(f"Duplicate mask basename in ZIP: {name}")
                archive_masks[name] = member

        for i, row in enumerate(val.itertuples(index=False), start=1):
            image_id = str(row.image_id)
            filename = Path(str(filenames[image_id])).name
            if filename not in archive_masks:
                raise FileNotFoundError(f"Mask missing for {image_id}: {filename}")
            with Image.open(args.images_root / f"{image_id}.jpg") as source:
                image = source.convert("RGB")
            with Image.open(BytesIO(archive.read(archive_masks[filename]))) as source:
                ref_mask = source.convert("L")
            if image.size != ref_mask.size:
                raise ValueError(f"Image/mask size mismatch for {image_id}")

            # Inference is intentionally identical to export_predicted_roi_fold5.py.
            resized = image.resize((256, 192), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(device)
            predicted = (torch.sigmoid(model(tensor))[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
            pred_mask = Image.fromarray(predicted * 255).resize(
                image.size, Image.Resampling.NEAREST
            )

            ref_geom = geometry(ref_mask)
            pred_geom = geometry(pred_mask)
            # Verify against the production crop function before saving findings.
            for mask, geom, name in ((ref_mask, ref_geom, "reference"),
                                     (pred_mask, pred_geom, "predicted")):
                cropped, actual_status = crop_lesion_with_margin(image, mask)
                if actual_status != geom["status"] or cropped.size != (
                    geom["crop_width"], geom["crop_height"]
                ):
                    raise AssertionError(f"ROI logic mismatch for {image_id} ({name})")

            dice, iou, reference_binary = mask_overlap(ref_mask, pred_mask)
            c = comparison.loc[image_id]
            result = {
                "fold": fold, "image_id": image_id, "dx": row.dx,
                "reference_prediction": int(c.reference_prediction),
                "predicted_mask_prediction": int(c.predicted_mask_prediction),
                "reference_class": CLASSES[int(c.reference_prediction)],
                "predicted_class": CLASSES[int(c.predicted_mask_prediction)],
                "outcome": c.outcome,
                "mask_dice_original_resolution": dice,
                "mask_iou_original_resolution": iou,
                "reference_mask_pixel_retention_by_predicted_crop": pixel_retention(reference_binary, pred_geom),
                "roi_transition": f"{ref_geom['status']} -> {pred_geom['status']}",
                "roi_vs_fallback_transition": (
                    ("roi" if ref_geom["status"] == "roi" else "fallback") + " -> " +
                    ("roi" if pred_geom["status"] == "roi" else "fallback")
                ),
                "crop_fraction_delta": pred_geom["crop_fraction"] - ref_geom["crop_fraction"],
                "image_width": image.width, "image_height": image.height,
            }
            add_prefix(result, "reference", ref_geom)
            add_prefix(result, "predicted", pred_geom)
            records.append(result)
            if i % 250 == 0 or i == len(val):
                print(f"Fold {fold}: diagnosed {i}/{len(val)}")

    df = pd.DataFrame(records)
    if len(df) != len(val) or not df.image_id.is_unique:
        raise AssertionError("Incorrect diagnostic result count/IDs")
    counts = df.outcome.value_counts().to_dict()
    statuses = df.predicted_status.value_counts().to_dict()
    transitions = (
        df.groupby(["roi_vs_fallback_transition", "outcome"])
        .size().rename("count").reset_index().to_dict("records")
    )
    summary = {
        "fold": fold, "n_images": int(len(df)),
        "outcome_counts": counts,
        "predicted_crop_status_counts": statuses,
        "outcomes_by_transition": transitions,
        "mean_mask_dice_original_resolution": float(df.mask_dice_original_resolution.mean()),
        "mean_reference_retention_for_predicted_roi": (
            float(df.loc[df.predicted_status == "roi", "reference_mask_pixel_retention_by_predicted_crop"].mean())
        ),
        "notes": [
            "Development-fold diagnostic; not independent clinical validation.",
            "Classification confidence is NOT in the saved prediction CSV and is not computed here.",
            "Mask Dice here is recomputed at original resolution; it can differ from validation Dice at 256x192.",
            "No classification or segmentation checkpoint was modified.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental report overwrite.
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        df.to_csv(handle, index=False)
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("\nPredicted crop statuses:", statuses)
    print("Prediction outcomes:", counts)
    print("\nROI/fallback transitions by outcome:")
    print(pd.DataFrame(transitions).to_string(index=False))
    print("\nSaved:", output_csv, "and", output_json)


if __name__ == "__main__":
    main()
