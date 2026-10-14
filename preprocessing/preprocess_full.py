from pathlib import Path
from preprocessing.preprocess_split_sample import preprocess_one_record
import json
from preprocessing.validate_image_paths import load_metadata
from preprocessing.split_config import build_split_config
from preprocessing.split_dataset import group_records_by_lesion_id, split_lesion_ids, assign_records_to_splits
from preprocessing.preprocessing_config import load_preprocessing_config


def preprocess_split(records, split_name, image_dir, output_dir, config):
    result = []
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    for record in records:
        preprocess = preprocess_one_record(record, split_name, image_dir, output_dir, config)
        result.append(preprocess)
    return result

def preprocess_all_splits(splits, image_dir, output_dir, config):
    result = {}
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    for split_name in ["train", "val", "test"]:
        records = splits[split_name]
        processed_records = preprocess_split(records, split_name, image_dir, output_dir, config)
        result[split_name] = processed_records
    return result

def build_full_preprocessing_report(manifest):
    train_count = len(manifest["train"])
    val_count = len(manifest["val"])
    test_count = len(manifest["test"])

    total_processed = train_count + val_count + test_count
    result = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "total_processed": total_processed
    }
    return result

def save_json(data, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")
    config = build_split_config()
    grouped = group_records_by_lesion_id(records)
    lesion_ids = list(grouped.keys())
    split_ids = split_lesion_ids(lesion_ids, config)
    splits = assign_records_to_splits(grouped, split_ids)
    preprocessing_config = load_preprocessing_config("preprocessing_config.json")
    image_dir = "data/images"
    output_dir = "data/processed"
    manifest = preprocess_all_splits(splits, image_dir, output_dir, preprocessing_config)
    data = build_full_preprocessing_report(manifest)
    save_json(data, "full_preprocessing_report.json")
    save_json(manifest, "full_preprocessing_manifest.json")
    print(data)
    

