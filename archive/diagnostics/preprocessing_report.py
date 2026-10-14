import json
from pathlib import Path
from inspect_images import inspect_image_sample
from validate_image_paths import load_metadata



def build_preprocessing_report(sample):
    total_checked = len(sample)
    all_rgb = True
    original_sizes = {}
    for record in sample:
        if record["mode"] != "RGB":
            all_rgb = False
        size = f'{record["width"]}x{record["height"]}'
        if size not in original_sizes:
            original_sizes[size] = 1
        else:
            original_sizes[size] += 1
            
    result = {
        "total_checked" : total_checked,
        "all_rgb" : all_rgb,
        "original_sizes" : original_sizes
    }
    result["ready_for_resize"] = is_sample_ready_for_resize(result)
    return result
    

def is_sample_ready_for_resize(report):
    original_key = "600x450"
    all_rgb = report["all_rgb"]
    original_len = len(report["original_sizes"]) == 1
    has_expected_size = original_key in report["original_sizes"]
    
    if all_rgb and original_len and has_expected_size:
       return True
    return False

def save_preprocessing_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)

if __name__ == "__main__":

    records = load_metadata("HAM10000_metadata.csv")
    sample = inspect_image_sample(records, "data/images", 5)
    report = build_preprocessing_report(sample)
    save_preprocessing_report(report, "preprocessing_report.json")
    print(report)
    