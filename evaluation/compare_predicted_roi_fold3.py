import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import CrossValidationDataset


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RUN = "convnext_tiny_natural_ce_roi_margin15_seed123"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_, val = get_fold_metadata(3)
val = val.reset_index(drop=True)

transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()

reference_dataset = CrossValidationDataset(
    val,
    "data/processed_roi_margin15_v1",
    transform=transform,
)

predicted_metadata = val.copy()
predicted_metadata["source_split"] = "val"

predicted_dataset = CrossValidationDataset(
    predicted_metadata,
    "data/processed_roi_predicted_fold3_v1",
    transform=transform,
)

assert reference_dataset.classes == CLASSES
assert predicted_dataset.classes == CLASSES
assert len(reference_dataset) == len(predicted_dataset) == 1706

for dataset in (reference_dataset, predicted_dataset):
    assert all(Path(path).is_file() for path, _ in dataset.samples)


model = convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(
    model.classifier[2].in_features, 7
)

checkpoint = torch.load(
    f"checkpoints/{RUN}_fold3_best.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"], strict=True
)
model = model.to(device).eval()


def evaluate(dataset):
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

    labels, predictions = [], []

    with torch.inference_mode():
        for images, targets in loader:
            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                logits = model(images.to(device))

            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())

    return np.array(labels), np.array(predictions)


y_ref, p_ref = evaluate(reference_dataset)
y_pred, p_pred = evaluate(predicted_dataset)

assert np.array_equal(y_ref, y_pred)

def macro_f1(predictions):
    return f1_score(
        y_ref, predictions,
        labels=list(range(7)),
        average="macro",
        zero_division=0,
    )


ref_f1 = macro_f1(p_ref)
pred_f1 = macro_f1(p_pred)

reference_report = json.loads(
    Path(f"reports/{RUN}_fold3_evaluation.json")
    .read_text(encoding="utf-8")
)

assert abs(
    ref_f1 - reference_report["macro_f1"]
) <= 0.003, "Reference checkpoint replay mismatch"

corrected = int(((p_ref != y_ref) & (p_pred == y_ref)).sum())
regressed = int(((p_ref == y_ref) & (p_pred != y_ref)).sum())

print("\nFold 3 — Same classifier, different ROI inputs")
print("Reference macro F1:", round(ref_f1, 4))
print("Predicted macro F1:", round(pred_f1, 4))
print("Macro F1 delta:", round(pred_f1 - ref_f1, 4))

print("\nReference accuracy:", round(accuracy_score(y_ref, p_ref), 4))
print("Predicted accuracy:", round(accuracy_score(y_ref, p_pred), 4))

print("\nPrediction changes:", int((p_ref != p_pred).sum()))
print("Errors corrected:", corrected)
print("New errors:", regressed)

print("\nPer-class F1:")
for idx, name in enumerate(CLASSES):
    ref_score = f1_score(
        y_ref, p_ref, labels=[idx],
        average=None, zero_division=0
    )[0]
    pred_score = f1_score(
        y_ref, p_pred, labels=[idx],
        average=None, zero_division=0
    )[0]

    print(
        f"{name}: reference={ref_score:.4f}, "
        f"predicted={pred_score:.4f}, "
        f"delta={pred_score - ref_score:+.4f}"
    )

output = Path(
    "reports/segmentation/small_unet_fold3_seed123/"
    "classifier_predicted_vs_reference_fold3.csv"
)

if output.exists():
    raise FileExistsError(output)

pd.DataFrame({
    "image_id": val["image_id"],
    "dx": val["dx"],
    "true_label": y_ref,
    "reference_prediction": p_ref,
    "predicted_mask_prediction": p_pred,
}).to_csv(output, index=False)

print("\nSaved comparison:", output)