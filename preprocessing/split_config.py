import json
from pathlib import Path

def build_split_config():
    return {
        "group_by" : "lesion_id",
        "train_ratio" : 0.7,
        "val_ratio" : 0.15,
        "test_ratio" : 0.15,
        "random_seed" : 42
    }

def validate_split_config(config):
    total_ratio = config["train_ratio"] + config["val_ratio"] + config["test_ratio"]

    condition_group = config["group_by"] == "lesion_id"
    condition_total = abs(total_ratio - 1.0) < 0.01
    condition_random_seed = isinstance(config["random_seed"], int)

    if condition_group and condition_total and condition_random_seed:
        return True

    return False

def save_split_config(config, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(config, file, indent=2)

if __name__ == "__main__":
    config = build_split_config()
    print(validate_split_config(config))
    save_split_config(config, "split_config.json")
    print(config)