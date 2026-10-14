"""Combine independent ConvNeXt and PanDerm out-of-fold probabilities by IMAGE ID.

Run from the Mole Classification project root, after both exports:
    python merge_model_oof_ensemble.py

Each model's probabilities must come from its OWN original validation fold.
Never assume both models have identical fold assignments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path("reports")
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load(name):
    path = ROOT / f"oof_{name}_original_folds.npz"
    with np.load(path, allow_pickle=False) as data:
        required = ("image_ids", "lesion_ids", "labels", "folds", "probabilities", "class_names")
        if not all(k in data.files for k in required):
            raise RuntimeError(f"Missing fields in {path}")
        return {k: data[k].copy() for k in required}


def metrics(y, p):
    pred = np.argmax(p, axis=1)
    indices = list(range(7))
    return {
        "count": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, labels=indices, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=indices, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=indices, average="macro", zero_division=0)),
        "per_class_f1": {name: float(score) for name, score in zip(
            CLASS_NAMES, f1_score(y, pred, labels=indices, average=None, zero_division=0))},
    }


def main():
    conv = load("convnext")
    pan = load("panderm")
    for name, obj in [("convnext", conv), ("panderm", pan)]:
        ids = obj["image_ids"].tolist()
        if len(ids) != 8530 or len(ids) != len(set(ids)):
            raise RuntimeError(f"{name}: expected 8530 unique image IDs")
        if obj["probabilities"].shape != (8530, 7):
            raise RuntimeError(f"{name}: probability shape not (8530, 7)")
        if obj["class_names"].tolist() != CLASS_NAMES:
            raise RuntimeError(f"{name}: class order mismatch")
        if not np.all((obj["folds"] >= 1) & (obj["folds"] <= 5)):
            raise RuntimeError(f"{name}: invalid fold indices")
    pan_pos = {image_id: i for i, image_id in enumerate(pan["image_ids"].tolist())}
    conv_ids = conv["image_ids"].tolist()
    if set(conv_ids) != set(pan_pos):
        raise RuntimeError("ConvNeXt and PanDerm OOF sets have different image IDs")
    idx = np.asarray([pan_pos[image_id] for image_id in conv_ids], dtype=np.int64)
    if not np.array_equal(conv["labels"], pan["labels"][idx]):
        raise RuntimeError("Image-ID matched labels disagree")
    if not np.array_equal(conv["lesion_ids"], pan["lesion_ids"][idx]):
        raise RuntimeError("Image-ID matched lesion IDs disagree")
    y = conv["labels"]
    pc = conv["probabilities"]
    pp = pan["probabilities"][idx]
    pe = (pc + pp) / 2.0
    fold_c, fold_p = conv["folds"], pan["folds"][idx]

    same_fold = int((fold_c == fold_p).sum())
    print(f"Verified 8530 matching image IDs and labels; same numeric fold: {same_fold}/8530")
    print("Each model's prediction was exported from its OWN held-out fold.")
    results = {"comparison_type": "pooled out-of-fold predictions; development data, NOT independent test",
               "total": 8530, "same_numeric_fold_count": same_fold,
               "convnext": metrics(y, pc), "panderm": metrics(y, pp), "ensemble_50_50": metrics(y, pe)}
    for name in ("convnext", "panderm", "ensemble_50_50"):
        m = results[name]
        print(f"{name:15s} accuracy={m['accuracy']:.4f} macro F1={m['macro_f1']:.4f}")
    pred_c, pred_p = pc.argmax(axis=1), pp.argmax(axis=1)
    results["complementarity"] = {
        "convnext_wrong_panderm_correct": int(((pred_c != y) & (pred_p == y)).sum()),
        "convnext_correct_panderm_wrong": int(((pred_c == y) & (pred_p != y)).sum()),
        "both_wrong": int(((pred_c != y) & (pred_p != y)).sum()),
    }
    results["by_convnext_fold"] = {}
    for fold in range(1, 6):
        subset = fold_c == fold
        results["by_convnext_fold"][str(fold)] = {
            "convnext": metrics(y[subset], pc[subset]),
            "panderm": metrics(y[subset], pp[subset]),
            "ensemble_50_50": metrics(y[subset], pe[subset]),
        }
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "oof_convnext_panderm_ensemble_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    np.savez_compressed(ROOT / "oof_convnext_panderm_ensemble_probabilities.npz",
                        image_ids=conv["image_ids"], lesion_ids=conv["lesion_ids"], labels=y,
                        convnext_fold=fold_c, panderm_fold=fold_p,
                        convnext=pc, panderm=pp, ensemble=pe)
    print("Saved reports/oof_convnext_panderm_ensemble_results.json")
    print("Important: pooled OOF F1 is not the arithmetic mean of five fold F1 values.")


if __name__ == "__main__":
    main()
