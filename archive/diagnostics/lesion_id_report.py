from validate_image_paths import load_metadata
import json
from pathlib import Path

def build_lesion_id_report(records):
    lesion_ids = set()
    duplicate_lesion_ids = set()
    for record in records:
        lesion_id = record["lesion_id"]
        if lesion_id in lesion_ids:
            duplicate_lesion_ids.add(lesion_id)
        else:
            lesion_ids.add(lesion_id)

    result = {
        "total_records": len(records),
        "unique_lesion_ids": len(lesion_ids),
        "duplicate_lesion_id_count": len(duplicate_lesion_ids),
        "extra_image_records": len(records) - len(lesion_ids)
     }

    return result

def save_lesion_id_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)

if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")
    report = build_lesion_id_report(records)
    save_lesion_id_report(report, "lesion_id_report.json")
    print(report)