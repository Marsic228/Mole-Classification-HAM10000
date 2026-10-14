from pprint import pprint
from archive.diagnostics.dataset_config import DatasetConfig
from archive.diagnostics.experiment_run import ExperimentRun
import json
import csv

def load_metadata_sample(path, limit):
    result = []
    with open(path , "r") as file:
        loaded = csv.DictReader(file)
        for i in loaded:
            if len(result) >= limit:
                break
            result.append(i)
    return result

def count_labels(records):
    label_counts = {}

    for record in records:
        label = record["dx"]

        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    return label_counts

def build_dataset_summary(records, config, experiment):
    summary = {}

    image_ids = []
    labels = []
    unique_labels = []
    for record in records:
        image_ids.append(record["image_id"])
        label = record["dx"]
        labels.append(label)
        if label not in unique_labels:
            unique_labels.append(label)

    summary["total_records"] = len(records)
    summary["has_records"] = len(records) > 0
    summary["class_count"] = config.num_classes
    summary["image_size"] = list(config.image_size)
    summary["config"] = config.to_dict()
    summary["experiment"] = experiment.to_dict()
    summary["unique_labels"] = unique_labels
    summary["label_counts"] = count_labels(records)

    return summary



def save_summary(summary, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    records = load_metadata_sample("HAM10000_metadata.csv", 10015)

    config = DatasetConfig(
        metadata_path="HAM10000_metadata.csv",
        image_dirs=["data/images"],
        num_classes=7,
        image_size=(224, 224),
        split_ratios=(0.7, 0.15, 0.15)
    )

    experiment = ExperimentRun(
        experiment_id="summary_test_001",
        model_name="no_model_yet",
        dataset_config_path="dataset_config.json",
        num_epochs=0,
        learning_rate=0.0
    )

    summary = build_dataset_summary(records, config, experiment)


    pprint(summary)
    save_summary(summary, "dataset_summary.json")

    print(len(records))
    print(records[0])

    test_records = [
    {"dx": "bkl"},
    {"dx": "nv"},
    {"dx": "bkl"}
]

    print(count_labels(test_records))