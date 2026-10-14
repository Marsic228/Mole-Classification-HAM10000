from pathlib import Path
from PIL import Image
from preprocessing.validate_image_paths import load_metadata
from preprocessing.split_config import build_split_config
from preprocessing.split_dataset import group_records_by_lesion_id, split_lesion_ids, assign_records_to_splits
from preprocessing.preprocessing_config import load_preprocessing_config
import json

def build_output_path(output_dir, split_name, label, image_id, config):
    width = config["resize_width"]
    height = config["resize_height"]
    output_path = Path(output_dir)/f"{split_name}"/f"{label}"/f"{image_id}_{width}x{height}.jpg"
    return output_path

def ensure_output_folder(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

def resize_and_save_image(input_path, output_path, config):
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = Image.open(input_path)
    target_size = (config["resize_width"], config["resize_height"])
    resized_image = image.resize(target_size)
    ensure_output_folder(output_path)
    resized_image.save(output_path)
    return output_path

def preprocess_one_record(record, split_name, image_dir, output_dir, config):
    image_id = record["image_id"]
    label = record["dx"]
    input_path = Path(image_dir)/f"{image_id}.jpg"
    output_path = build_output_path(output_dir, split_name, label, image_id, config)
    saved_path = resize_and_save_image(input_path, output_path, config)
    result = {
        "image_id":image_id,
        "split": split_name,
        "label": label,
        "saved_path": str(saved_path)
    }
    return result

def preprocess_split_sample(records, split_name, image_dir, output_dir, config, limit):
    result = []
    for record in records:
        if len(result) >= limit:
            break
        preprocessed_record = preprocess_one_record(record, split_name, image_dir, output_dir, config)
        result.append(preprocessed_record)
    return result

def preprocess_all_split_samples(splits, image_dir, output_dir, config, limit_per_split):
    result = {}
    for split_name in ["train", "val", "test"]:
        records = splits[split_name]
        preprocess_split = preprocess_split_sample(records, split_name, image_dir, output_dir, config, limit_per_split)
        result[split_name] = preprocess_split
    return result

def save_preprocessing_manifest(manifest, path):
    path = Path(path)
    with open(path, "w",encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

def build_preprocessing_sample_report(manifest):
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

def save_preprocessing_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)

def validate_processed_image(image_path, expected_width, expected_height):
    image_path = Path(image_path)
    exists = False
    size_ok = False
    width = None
    height = None
    if image_path.exists():
        exists = True
        image = Image.open(image_path)
        width = image.width
        height = image.height
    if width == expected_width and height == expected_height:
        size_ok = True
    result = {
        "path": str(image_path),
        "exists": exists,
        "width": width,
        "height": height,
        "size_ok": size_ok
    }
    return result

def validate_processed_manifest(manifest, expected_width, expected_height):
    result = {}
    for split_name in ["train", "val", "test"]:
        records = manifest[split_name]
        result[split_name] = []
        for record in records:
            image_path = record["saved_path"]
            validation = validate_processed_image(image_path, expected_width, expected_height)
            result[split_name].append(validation)
    return result

def build_processed_validation_report(validation_manifest):
    missing_count = 0
    wrong_size_count = 0
    all_files_exist = True
    all_sizes_ok = True
    train_count = len(validation_manifest["train"])
    val_count = len(validation_manifest["val"])
    test_count = len(validation_manifest["test"])
    total_checked = train_count + val_count + test_count
    for split_name in ["train", "val", "test"]:
        records = validation_manifest[split_name]
        for record in records:
            if record["size_ok"] == False:
                wrong_size_count += 1
                all_sizes_ok = False
            if record["exists"] == False:
                all_files_exist = False
                missing_count += 1
    result = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "total_checked": total_checked,
        "missing_count": missing_count,
        "wrong_size_count": wrong_size_count,
        "all_files_exist": all_files_exist,
        "all_sizes_ok": all_sizes_ok
    }
    return result

def save_processed_validation_report(report, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
            

if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")

    split_config = build_split_config()
    grouped = group_records_by_lesion_id(records)
    lesion_ids = list(grouped.keys())
    split_ids = split_lesion_ids(lesion_ids, split_config)
    splits = assign_records_to_splits(grouped, split_ids)

    image_dir = "data/images"
    output_dir = "data/processed_sample_real"
    preprocessing_config = load_preprocessing_config("preprocessing_config.json")
    limit_per_split = 2

    preprocessing = preprocess_all_split_samples(
        splits,
        image_dir,
        output_dir,
        preprocessing_config,
        limit_per_split
    )
    save_preprocessing_manifest(preprocessing, "preprocessing_sample_manifest.json")

    report = build_preprocessing_sample_report(preprocessing)
    save_preprocessing_report(report, "preprocessing_sample_report.json")
    validation_manifest = validate_processed_manifest(preprocessing,
    preprocessing_config["resize_width"],
    preprocessing_config["resize_height"]
    )

    validation_report = build_processed_validation_report(validation_manifest)

    save_preprocessing_report(validation_report, "processed_sample_validation_report.json")
    print(validation_report)

