from pathlib import Path
import json
from validate_image_paths import load_metadata
from split_config import build_split_config
from split_dataset import group_records_by_lesion_id, split_lesion_ids, assign_records_to_splits

def count_labels(records):
    result = {}
    for record in records:
        label = record["dx"]

        if label not in result:
            result[label] = 1
        else:
            result[label] += 1
    return result

def build_split_balance_report(splits):
    result = {}

    for split_name in ["train", "val", "test"]:
        records = splits[split_name]

        result[split_name] = {
            "total_records": len(records),
            "label_distribution": build_label_distribution(records)
        }

    return result

def build_label_distribution(records):
    label_counts = count_labels(records)
    total_records = len(records)
    if total_records == 0:
            return{}
    result = {}
    for label, count in label_counts.items():
        percent = count / total_records * 100
        percent = round(percent, 2)

        result[label] = {
            "count": count,
            "percent": percent
        }

    return result

        

def save_split_balance_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)
        

if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")
    config = build_split_config()
    grouped = group_records_by_lesion_id(records)
    lesion_ids = list(grouped.keys())
    split_ids = split_lesion_ids(lesion_ids, config)
    splits = assign_records_to_splits(grouped, split_ids)
    report = build_split_balance_report(splits)
    save_split_balance_report(report, "split_balance_report.json")
    print(report)







    
