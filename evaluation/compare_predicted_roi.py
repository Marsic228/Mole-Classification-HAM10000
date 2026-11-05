"""Replay D22 reference and predicted ROI inputs without overwriting evidence."""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RUN = "convnext_tiny_natural_ce_roi_margin15_seed123"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    for name in ("checkpoint", "reference-root", "predicted-root", "reference-report", "output-root"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_paths(args):
    n = args.fold
    defaults = {
        "checkpoint": f"checkpoints/{RUN}_fold{n}_best.pt",
        "reference_root": "data/processed_roi_margin15_v1",
        "predicted_root": f"data/processed_roi_predicted_fold{n}_v1",
        "reference_report": f"reports/{RUN}_fold{n}_evaluation.json",
        "output_root": f"reports/segmentation/small_unet_fold{n}_seed123",
    }
    paths = {k: (REPO_ROOT / (getattr(args, k) or v)).resolve() for k, v in defaults.items()}
    paths["output"] = paths["output_root"] / f"classifier_predicted_vs_reference_fold{n}.csv"
    return paths


def verify_checkpoint(path, fold):
    historical = f"checkpoints/{RUN}_fold{fold}_best.pt"
    manifest = json.loads((REPO_ROOT / "reports/segmentation/phase1_artifact_manifest.json").read_text(encoding="utf-8"))
    entry = next(a for a in manifest["artifacts"] if a["path"] == historical)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Checkpoint does not match the historical fold {fold} fingerprint: {path}")


def load_metadata(fold):
    from core.canonical_cv import get_fold_metadata
    with chdir(REPO_ROOT):
        _, val = get_fold_metadata(fold)
    return val.reset_index(drop=True)


def validate_pairs(val, reference, predicted):
    if reference.classes != CLASSES or predicted.classes != CLASSES:
        raise ValueError("Dataset class order differs from the historical seven-class order")
    if val["image_id"].duplicated().any() or len(reference) != len(val) or len(predicted) != len(val):
        raise ValueError("Duplicate IDs or mismatched dataset lengths")
    for i, row in enumerate(val.itertuples(index=False)):
        expected = CLASSES.index(row.dx)
        for dataset in (reference, predicted):
            path, label = dataset.samples[i]
            if label != expected or Path(path).name != f"{row.image_id}_224x224.jpg":
                raise ValueError(f"Image/label ordering mismatch at {row.image_id}")
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing classifier image: {path}")


def preflight(args, paths):
    if paths["output"].exists():
        raise FileExistsError(f"Output exists; select a fresh --output-root: {paths['output']}")
    if paths["output_root"].exists() and not paths["output_root"].is_dir():
        raise ValueError("--output-root must be a directory")
    for name in ("checkpoint", "reference_report"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    verify_checkpoint(paths["checkpoint"], args.fold)
    from core.cross_validation_dataset import CrossValidationDataset
    val = load_metadata(args.fold)
    reference = CrossValidationDataset(val, paths["reference_root"])
    predicted_metadata = val.copy()
    predicted_metadata["source_split"] = "val"
    predicted = CrossValidationDataset(predicted_metadata, paths["predicted_root"])
    validate_pairs(val, reference, predicted)
    report = json.loads(paths["reference_report"].read_text(encoding="utf-8"))
    if report["fold"] != args.fold or report["class_names"] != CLASSES or report["total_checked"] != len(val):
        raise ValueError("Reference report fold, classes or image count mismatch")
    return val, reference, predicted, report


def select_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def build_model(checkpoint_path, device):
    import torch
    import torch.nn as nn
    from torchvision.models import convnext_tiny
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 7)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def evaluate(dataset, model, device, *, capture_probabilities=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    labels, predictions, probabilities = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images.to(device))
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            if capture_probabilities:
                probabilities.extend(logits.float().softmax(1).cpu().tolist())
    result = (np.array(labels), np.array(predictions))
    return (*result, np.array(probabilities)) if capture_probabilities else result


def macro_f1(labels, predictions):
    from sklearn.metrics import f1_score
    return f1_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)


def run(args, paths, checked):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from torchvision.models import ConvNeXt_Tiny_Weights
    val, reference, predicted, report = checked
    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    reference.transform = predicted.transform = transform
    device = select_device(args.device)
    model = build_model(paths["checkpoint"], device)
    y_ref, p_ref = evaluate(reference, model, device)
    y_pred, p_pred = evaluate(predicted, model, device)
    if not np.array_equal(y_ref, y_pred):
        raise ValueError("Reference/predicted labels differ")
    ref_f1, pred_f1 = macro_f1(y_ref, p_ref), macro_f1(y_ref, p_pred)
    if not abs(ref_f1 - report["macro_f1"]) <= 0.003:
        raise ValueError("Reference checkpoint replay mismatch")
    print(f"Fold {args.fold} — Same classifier, different ROI inputs")
    print("Reference macro F1:", round(ref_f1, 4))
    print("Predicted macro F1:", round(pred_f1, 4))
    print("Macro F1 delta:", round(pred_f1 - ref_f1, 4))
    print("Reference accuracy:", round(accuracy_score(y_ref, p_ref), 4))
    print("Predicted accuracy:", round(accuracy_score(y_ref, p_pred), 4))
    print("Prediction changes:", int((p_ref != p_pred).sum()))
    print("Errors corrected:", int(((p_ref != y_ref) & (p_pred == y_ref)).sum()))
    print("New errors:", int(((p_ref == y_ref) & (p_pred != y_ref)).sum()))
    print("Per-class F1:")
    for idx, name in enumerate(CLASSES):
        ref_score = f1_score(y_ref, p_ref, labels=[idx], average=None, zero_division=0)[0]
        pred_score = f1_score(y_ref, p_pred, labels=[idx], average=None, zero_division=0)[0]
        print(f"{name}: reference={ref_score:.4f}, predicted={pred_score:.4f}, delta={pred_score-ref_score:+.4f}")
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    with paths["output"].open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"image_id": val["image_id"], "dx": val["dx"], "true_label": y_ref,
                      "reference_prediction": p_ref, "predicted_mask_prediction": p_pred}).to_csv(handle, index=False)
    print("Saved comparison:", paths["output"])


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        paths = resolve_paths(args)
        checked = preflight(args, paths)
        select_device(args.device)
        if args.dry_run:
            print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
            print(f"Validated {len(checked[0])} ordered images; dry-run performed no writes or inference.")
            return
        run(args, paths, checked)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        p.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
