import json
from pathlib import Path


report_paths = [
    Path("reports/resnet18_focal_mel_confusion_fold1_epoch_3_evaluation.json"),
    Path("reports/resnet18_focal_mel_confusion_fold2_epoch_3_evaluation.json"),
    Path("reports/resnet18_focal_mel_confusion_fold3_epoch_3_evaluation.json"),
    Path("reports/resnet18_focal_mel_confusion_fold4_epoch_3_evaluation.json"),
    Path("reports/resnet18_focal_mel_confusion_fold5_epoch_3_evaluation.json"),
]


# 2. Sum the five confusion matrices

class_names = None
total_matrix = None

for path in report_paths:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    matrix = data["confusion_matrix"]

    if class_names is None:
        class_names = data["class_names"]
        size = len(class_names)
        total_matrix = [
            [0 for _ in range(size)]
            for _ in range(size)
        ]

    for row_index in range(len(matrix)):
        for col_index in range(len(matrix[row_index])):
            total_matrix[row_index][col_index] += matrix[row_index][col_index]

mel_index = class_names.index("mel")

mel_row = total_matrix[mel_index]

print("True MEL predicted as:")
for class_name, count in zip(class_names, mel_row):
    print(class_name, count)