from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from core.canonical_cv import get_fold_metadata

root = Path("reports/segmentation/small_unet_fold1_seed123")

pred = pd.read_csv(
    root / "classifier_predicted_vs_reference_fold1.csv"
)

roi = pd.read_csv(
    root / "roi_retention_audit_original_resolution.csv"
)

df = pred.merge(
    roi[[
        "image_id", "reference_status", "status",
        "dice", "retention", "crop_fraction"
    ]],
    on="image_id",
    validate="one_to_one",
)

assert len(df) == 1707

reference_correct = df["reference_prediction"] == df["true_label"]
predicted_correct = df["predicted_mask_prediction"] == df["true_label"]

df["changed"] = (
    df["reference_prediction"] != df["predicted_mask_prediction"]
)
df["corrected"] = ~reference_correct & predicted_correct
df["regressed"] = reference_correct & ~predicted_correct
df["wrong_to_wrong"] = (
    df["changed"] & ~reference_correct & ~predicted_correct
)

def category(status):
    return "ROI" if status == "roi" else "fallback"

df["transition"] = (
    df["reference_status"].map(category)
    + " -> "
    + df["status"].map(category)
)

print("\nClassification changes by crop transition:")

print(
    df.groupby("transition")
    .agg(
        images=("image_id", "size"),
        changed=("changed", "sum"),
        corrected=("corrected", "sum"),
        regressed=("regressed", "sum"),
        wrong_to_wrong=("wrong_to_wrong", "sum"),
    )
    .to_string()
)

print("\nRegressions with lowest lesion retention:")

print(
    df[df["regressed"]]
    .sort_values("retention")
    [[
        "image_id", "dx", "transition",
        "dice", "retention", "crop_fraction"
    ]]
    .head(12)
    .to_string(index=False)
)

_, val = get_fold_metadata(1)

source_by_id = val.set_index("image_id")["source_split"].to_dict()

groups = [
    ("Corrected: ROI -> ROI",
     df[df["corrected"] & (df["transition"] == "ROI -> ROI")]),

    ("Regressed: ROI -> ROI",
     df[df["regressed"] & (df["transition"] == "ROI -> ROI")]),

    ("Regressed: ROI -> fallback",
     df[df["regressed"] & (df["transition"] == "ROI -> fallback")]),
]

cases = [
    (group_name, row)
    for group_name, subset in groups
    for _, row in subset.nsmallest(2, "dice").iterrows()
]

fig, axes = plt.subplots(len(cases), 3, figsize=(12, 18))

for i, (group_name, row) in enumerate(cases):
    image_id = row["image_id"]
    filename = f"{image_id}_224x224.jpg"

    paths = [
        Path("data/images") / f"{image_id}.jpg",
        Path("data/processed_roi_margin15_v1")
        / source_by_id[image_id] / row["dx"] / filename,
        Path("data/processed_roi_predicted_fold1_v1")
        / "val" / row["dx"] / filename,
    ]

    for j, path in enumerate(paths):
        with Image.open(path) as image:
            axes[i, j].imshow(image.convert("RGB"))

        axes[i, j].axis("off")

    for j, title in enumerate([
        f"{image_id} ({row['dx']})",
        "Reference ROI",
        "Predicted ROI",
    ]):
        axes[i, j].set_title(title, fontsize=10)

    axes[i, 0].set_ylabel(group_name, fontsize=9)

plt.tight_layout()
plt.show()