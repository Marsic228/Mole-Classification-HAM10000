from pathlib import Path
import csv

def load_metadata(path, limit=None):
    result = []
    with open(path, "r") as file:
        loaded = csv.DictReader(file) 
        for record in loaded:
            if limit is not None and len(result) >= limit:
                break
            result.append(record)
        return result

def find_missing_images(records, image_dir):
    missing = []
    image_dir = Path(image_dir)
    for record in records:
        image_id = record["image_id"]
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            missing.append(record)
    return missing

def summarize_missing_images(missing_records):
    missing_count = len(missing_records)
    missing_by_label = {}
    
    for record in missing_records:
        label = record["dx"]

        if label not in missing_by_label:
            missing_by_label[label] = 1
        else:
            missing_by_label[label] += 1

    result = {
        "missing_count" : missing_count,
        "missing_by_label" : missing_by_label
    }
    return result

    

if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")
    missing_records = find_missing_images(records, "data/images")
    summary = summarize_missing_images(missing_records)

    print(summary)
    if summary["missing_count"] == len(records):
        print("Warning: all images are missing.")
        print("Check whether the image folder exists and contains HAM10000 .jpg files.")

    