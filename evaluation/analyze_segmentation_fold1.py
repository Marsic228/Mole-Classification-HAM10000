import pandas as pd
from core.canonical_cv import get_fold_metadata

path = (
    "reports/segmentation/small_unet_fold1_seed123/"
    "best_validation_per_image.csv"
)

results = pd.read_csv(path)

_, val = get_fold_metadata(1)

df = results.merge(
    val[["image_id", "dx"]],
    on="image_id",
    validate="one_to_one",
)

assert len(df) == len(val) == 1707
assert set(df["image_id"]) == set(val["image_id"])

print("\nDice distribution:")
print(df["dice"].quantile(
    [0.01, 0.05, 0.10, 0.50, 0.90]
).round(4))

for threshold in (0.5, 0.7, 0.8):
    count = (df["dice"] < threshold).sum()
    print(f"Dice < {threshold}: {count} images")

print("\nSegmentation quality by diagnosis:")
print(
    df.groupby("dx")
    .agg(
        n=("dice", "size"),
        mean_dice=("dice", "mean"),
        mean_iou=("iou", "mean"),
    )
    .round(4)
)

print("\nWorst 12 predictions:")
print(
    df.nsmallest(12, "dice")[
        ["image_id", "dx", "dice", "iou", "predicted_empty"]
    ].to_string(index=False)
)