"""Train the historical SmallUNet configuration on one frozen canonical fold."""
import argparse
from contextlib import chdir
import csv
import json
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    for name in ("image-root", "mask-mapping", "mask-archive", "output-root"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_paths(args):
    defaults = {"image_root": "data/images", "mask_mapping": "Masks_to_IsicId_mapping.csv",
                "mask_archive": "Masks.zip", "output_root": "."}
    paths = {k: (REPO_ROOT / (getattr(args, k) or v)).resolve() for k, v in defaults.items()}
    paths["report_dir"] = paths["output_root"] / f"reports/segmentation/small_unet_fold{args.fold}_seed123"
    paths["checkpoint"] = paths["output_root"] / f"checkpoints/segmentation/small_unet_fold{args.fold}_seed123_best.pt"
    return paths


def preflight(args, paths):
    for name in ("report_dir", "checkpoint"):
        if paths[name].exists():
            raise FileExistsError(f"Experiment output exists; select a fresh --output-root: {paths[name]}")
    for name in ("mask_mapping", "mask_archive"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    from core.canonical_cv import get_fold_metadata
    with chdir(REPO_ROOT):
        train, val = get_fold_metadata(args.fold)
    with paths["mask_mapping"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"ISIC_ID", "Filename"}.issubset(reader.fieldnames or []):
            raise ValueError("Mask mapping requires ISIC_ID and Filename columns")
        mapping = {}
        for row in reader:
            if not row["ISIC_ID"] or row["ISIC_ID"] in mapping or not row["Filename"]:
                raise ValueError("Blank or duplicate mask mapping")
            mapping[row["ISIC_ID"]] = Path(row["Filename"]).name
    with ZipFile(paths["mask_archive"]) as archive:
        names = [Path(n).name for n in archive.namelist() if n.lower().endswith(".png")]
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous duplicate PNG basenames in mask archive")
    available = set(names)
    for metadata in (train, val):
        if metadata["image_id"].duplicated().any():
            raise ValueError("Duplicate canonical image IDs")
        for image_id in metadata["image_id"]:
            if not (paths["image_root"] / f"{image_id}.jpg").is_file():
                raise FileNotFoundError(f"Missing raw image: {image_id}")
            if image_id not in mapping or mapping[image_id] not in available:
                raise ValueError(f"Missing mapped mask for {image_id}")
    return len(train), len(val)


def select_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def initialize(args, paths):
    # Keep the original seed -> metadata -> mapping -> datasets -> loaders -> model -> Adam sequence.
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from core.canonical_cv import get_fold_metadata
    from core.cross_validation_dataset import SegmentationDataset
    from core.segmentation_unet import SmallUNet
    from core.segmentation_losses import segmentation_loss
    from evaluation.segmentation_validation import evaluate_segmentation
    torch.manual_seed(123)
    with chdir(REPO_ROOT):
        train, val = get_fold_metadata(args.fold)
    mapping = pd.read_csv(paths["mask_mapping"]).set_index("ISIC_ID")["Filename"]
    train_dataset = SegmentationDataset(train, mapping, image_root=paths["image_root"], mask_archive=paths["mask_archive"])
    val_dataset = SegmentationDataset(val, mapping, image_root=paths["image_root"], mask_archive=paths["mask_archive"])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    device = select_device(args.device)
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, train_loader, val_loader, val_dataset, device


def train_one_epoch(model, loader, optimizer, device):
    import torch
    from core.segmentation_losses import segmentation_loss
    model.train()

    total_loss = 0.0
    total_images = 0

    for batch_idx, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = segmentation_loss(logits, masks)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

        if batch_idx % 250 == 0 or batch_idx == len(loader):
            print(
                f"Batch {batch_idx}/{len(loader)} | "
                f"mean loss: {total_loss / total_images:.4f}"
            )

    return total_loss / total_images


def fit(model, optimizer, train_loader, val_loader, val_dataset, device, report_dir, checkpoint_path, fold):
    import pandas as pd
    import torch
    from time import perf_counter
    from evaluation.segmentation_validation import evaluate_segmentation
    history_path = report_dir / "history.json"
    best_dice = -1.0
    best_epoch = None
    epochs_without_improvement = 0

    history = []
    start = perf_counter()

    for epoch in range(1, 13):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )

        summary, records = evaluate_segmentation(
            model, val_loader, device
        )

        if summary["n_images"] != len(val_dataset) or len({r["image_id"] for r in records}) != len(val_dataset):
            raise ValueError("Validation count or unique IDs differ from canonical validation length")

        current_dice = summary["mean_dice"]

        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss,
            **summary,
        }
        history.append(epoch_report)

        print(
            f"\nEpoch {epoch}: "
            f"loss={train_loss:.4f}, "
            f"Dice={current_dice:.4f}, "
            f"IoU={summary['mean_iou']:.4f}, "
            f"empty={summary['empty_predictions']}"
        )

        if current_dice > best_dice + 1e-4:
            best_dice = current_dice
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "fold": fold,
                    "seed": 123,
                    "mean_dice": best_dice,
                    "config": {
                        "input_size": [192, 256],
                        "batch_size": 4,
                        "learning_rate": 1e-3,
                        "loss": "BCE + soft Dice",
                        "threshold": 0.5,
                    },
                },
                checkpoint_path,
            )

            pd.DataFrame(records).to_csv(
                report_dir / "best_validation_per_image.csv",
                index=False,
            )

            print("New best checkpoint saved.")
        else:
            epochs_without_improvement += 1

        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if epochs_without_improvement >= 4:
            print("Early stopping triggered.")
            break

    print(f"\nFinal Fold {fold} summary:")
    print("Best epoch:", best_epoch)
    print("Best Dice:", round(best_dice, 4))
    print("Elapsed minutes:", round((perf_counter() - start) / 60, 2))
    return best_epoch, best_dice


def run(args, paths):
    state = initialize(args, paths)
    # Exclusive report directory reserves this experiment; checkpoint is rechecked before writes.
    if paths["checkpoint"].exists():
        raise FileExistsError(paths["checkpoint"])
    paths["report_dir"].mkdir(parents=True, exist_ok=False)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    return fit(*state, paths["report_dir"], paths["checkpoint"], args.fold)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        paths = resolve_paths(args)
        counts = preflight(args, paths)
        select_device(args.device)
        if args.dry_run:
            print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
            print(f"Validated train={counts[0]}, val={counts[1]}; dry-run performed no writes or model construction.")
            return
        run(args, paths)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
