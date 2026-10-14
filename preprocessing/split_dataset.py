import random
from preprocessing.validate_image_paths import load_metadata
from preprocessing.split_config import build_split_config
import json
from pathlib import Path

def group_records_by_lesion_id(records):
    result = {}
    for record in records:
        lesion_id = record["lesion_id"]

        if lesion_id not in result:
            result[lesion_id] = []
        
        result[lesion_id].append(record)
    return result

def split_lesion_ids(lesion_ids, config):
    shuffled_ids = lesion_ids.copy()
    random.seed(config["random_seed"])
    random.shuffle(shuffled_ids)
    total = len(shuffled_ids)
    train_count = int(total * config["train_ratio"])
    val_count = int(total * config["val_ratio"])

    train_ids = shuffled_ids[:train_count]
    val_ids = shuffled_ids[train_count:train_count + val_count]
    test_ids = shuffled_ids[train_count + val_count:]

    result = {
        "train" : train_ids,
        "val" : val_ids,
        "test" : test_ids
    }
    return result

def assign_records_to_splits(grouped_records, split_lesion_ids):
    result = {
        "train" : [],
        "val" : [],
        "test" : []
    }

    for split_name in split_lesion_ids:
        lesion_ids = split_lesion_ids[split_name]

        for lesion_id in lesion_ids:
            records = grouped_records[lesion_id]
            result[split_name].extend(records)

    return result

def check_lesion_id_overlap(splits):
    train_ids = set()
    val_ids = set()
    test_ids = set()
    for record in splits["train"]:
        train_ids.add(record["lesion_id"])
    for record in splits["val"]:
        val_ids.add(record["lesion_id"])
    for record in splits["test"]:
        test_ids.add(record["lesion_id"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    is_leakage_safe = (
        len(train_test_overlap) == 0 
        and len(train_val_overlap) == 0 and
        len(val_test_overlap) == 0
    )

    result = {
        "train_val_overlap" : len(train_val_overlap),
        "train_test_overlap" : len(train_test_overlap),
        "val_test_overlap" : len(val_test_overlap),
        "is_leakage_safe" : is_leakage_safe
    }

    return result
        
def build_split_report(splits, leakage_report):
    return {
        "train_records" : len(splits["train"]),
        "val_records": len(splits["val"]),
        "test_records" : len(splits["test"]),
        "total_records" : len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
        "leakage_report": leakage_report
    }

def save_split_report(report, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

   
if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")

    config = build_split_config()

    grouped = group_records_by_lesion_id(records)
    lesion_ids = list(grouped.keys())
    split_ids = split_lesion_ids(lesion_ids, config)
    splits = assign_records_to_splits(grouped, split_ids)
    leakage_report = check_lesion_id_overlap(splits)
    split_report = build_split_report(splits, leakage_report)
    save_split_report(split_report, "split_dataset_report.json")
    print(split_report)  
        