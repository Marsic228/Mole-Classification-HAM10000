import csv
import json

DATA_FILE = "HAM10000_metadata.csv"
CLASS_MAP_FILE = "class_map.json"


def load_records(path):
    records = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
            return records
    except FileNotFoundError:
        return []

def load_class_map(path):
    try:
        with open(path) as f:
            loaded = json.load(f)
        return loaded
    except FileNotFoundError:
        return {}


def filter_by_label(records, target_label):
    result = []
    for i in records:
        if i["dx"] == target_label:
            result.append(i)
    return result

def main():
    records = load_records(DATA_FILE)
    print(f"Total records: {len(records)}")

    class_map = load_class_map(CLASS_MAP_FILE)

    for label in class_map:
        filtered = filter_by_label(records, label)
        print(f"{label} ({class_map[label]}): {len(filtered)} records")
    

if __name__ == "__main__":
    main()