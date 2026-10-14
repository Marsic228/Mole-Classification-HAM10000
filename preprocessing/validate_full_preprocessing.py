import json
from pathlib import Path
from preprocessing.preprocess_split_sample import validate_processed_image

def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded

def validate_full_manifest(manifest, expected_width, expected_height):
    result = {}
    for sample_name in ["train", "val", "test"]:
        records = manifest[sample_name]
        result[sample_name] =  []
        for record in records:
            image_path = record["saved_path"]
            validation = validate_processed_image(image_path, expected_width, expected_height)
            result[sample_name].append(validation)
    return result

def build_full_validation_report(validation_manifest):
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

def save_json(data, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    manifest = load_json("full_preprocessing_manifest.json")

    validation_manifest = validate_full_manifest(manifest, 224, 224)

    report = build_full_validation_report(validation_manifest)
    save_json(report, "full_preprocessing_validation_report.json")