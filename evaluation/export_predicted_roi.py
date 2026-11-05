"""Export canonical held-out predicted ROI crops, preserving historical encoding."""
import argparse
from collections import Counter
from contextlib import chdir
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    for name in ("checkpoint", "image-root", "output-root"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_paths(args):
    defaults = {"checkpoint": f"checkpoints/segmentation/small_unet_fold{args.fold}_seed123_best.pt",
                "image_root": "data/images", "output_root": f"data/processed_roi_predicted_fold{args.fold}_v1"}
    return {k: (REPO_ROOT / (getattr(args, k) or v)).resolve() for k, v in defaults.items()}


def preflight(args, paths):
    if paths["output_root"].exists():
        raise FileExistsError(f"Output root exists, even if empty; choose a fresh --output-root: {paths['output_root']}")
    checkpoint = paths["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing segmentation checkpoint: {checkpoint}")
    manifest = json.loads((REPO_ROOT / "reports/segmentation/phase1_artifact_manifest.json").read_text(encoding="utf-8"))
    historical = f"checkpoints/segmentation/small_unet_fold{args.fold}_seed123_best.pt"
    entry = next(a for a in manifest["artifacts"] if a["path"] == historical)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if checkpoint.stat().st_size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Checkpoint does not match historical fold {args.fold} fingerprint")
    from core.canonical_cv import get_fold_metadata, verified_assignments
    with chdir(REPO_ROOT):
        _, val = get_fold_metadata(args.fold)
        _, counts = verified_assignments()
    if val["image_id"].duplicated().any() or len(val) != counts[args.fold]["val"]:
        raise ValueError("Canonical validation IDs/counts are inconsistent")
    for row in val.itertuples(index=False):
        path = paths["image_root"] / f"{row.image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw image: {path}")
    return val


def select_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def build_model(checkpoint_path, device):
    import torch
    from core.segmentation_unet import SmallUNet
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_crop(image, model, device):
    import numpy as np
    import torch
    from PIL import Image
    from preprocessing.roi_crop import crop_lesion_with_margin
    resized = image.resize((256, 192), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(device)
    predicted = (torch.sigmoid(model(tensor))[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
    mask = Image.fromarray(predicted * 255).resize(image.size, Image.Resampling.NEAREST)
    cropped, status = crop_lesion_with_margin(image, mask)
    return cropped, status, predicted, mask


def validate_outputs(output_root, val):
    expected = {Path("val") / row.dx / f"{row.image_id}_224x224.jpg" for row in val.itertuples(index=False)}
    actual = {p.relative_to(output_root) for p in output_root.rglob("*") if p.is_file()}
    ids = [p.stem.removesuffix("_224x224") for p in actual]
    if len(expected) != len(val) or actual != expected or len(ids) != len(set(ids)):
        raise ValueError("Exported files differ from exact canonical IDs, counts or class paths")


def run(args, paths, val):
    import torch
    from PIL import Image
    device = select_device(args.device)
    model = build_model(paths["checkpoint"], device)
    paths["output_root"].mkdir(parents=True, exist_ok=False)
    status_counts = Counter()
    with torch.inference_mode():
        for row in val.itertuples(index=False):
            with Image.open(paths["image_root"] / f"{row.image_id}.jpg") as source:
                image = source.convert("RGB")
            cropped, status, _, _ = predict_crop(image, model, device)
            destination = paths["output_root"] / "val" / row.dx / f"{row.image_id}_224x224.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            cropped.resize((224, 224)).save(destination)
            status_counts[status] += 1
    validate_outputs(paths["output_root"], val)
    if sum(status_counts.values()) != len(val):
        raise ValueError("Export count differs from canonical validation length")
    print("Exported:", sum(status_counts.values()))
    print("Crop statuses:", dict(status_counts))
    return status_counts


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        paths = resolve_paths(args)
        val = preflight(args, paths)
        select_device(args.device)
        if args.dry_run:
            print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
            print(f"Validated {len(val)} ordered images; dry-run performed no writes or inference.")
            return
        run(args, paths, val)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        p.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
