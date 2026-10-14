import json
from inspect_images import inspect_one_image

def build_preprocessing_config():
    return {
        "expected_mode": "RGB",
        "original_width": 600,
        "original_height": 450,
        "resize_width": 224,
        "resize_height": 224,
        "normalize": False
    }

def validate_preprocessing_config(config):
    result = False
    mode_ok = config["expected_mode"] == "RGB"
    width_ok = config["resize_width"] > 0
    height_ok = config["resize_height"] > 0
    normalize_ok = isinstance(config["normalize"], bool) 
    if mode_ok and width_ok and height_ok and normalize_ok:
        result = True
    return result

def save_preprocessing_config(config, path):
    with open(path , "w") as file:
        json.dump(config, file, indent=2)

def load_preprocessing_config(path):
    with open(path, "r") as file:
        loaded = json.load(file)
    return loaded

def validate_image_against_config(image_info, config):
    result = False
    mode_check = image_info["mode"] == config["expected_mode"]
    width_check = image_info["width"] == config["original_width"]
    height_check = image_info["height"] == config["original_height"]
    if mode_check and width_check and height_check:
        result = True
    return result


if __name__ == "__main__":
    config = build_preprocessing_config()
    print("Config valid:", validate_preprocessing_config(config))
    save_preprocessing_config(config, "preprocessing_config.json")
    loaded_config = load_preprocessing_config("preprocessing_config.json")
    print("Loaded config:", validate_preprocessing_config(loaded_config))
    bad_config = build_preprocessing_config()
    bad_config["resize_width"] = 0
    print("Bad config:", validate_preprocessing_config(bad_config))
    real_image_info = inspect_one_image("data/images/ISIC_0024306.jpg")
    print("Real image valid:", validate_image_against_config(real_image_info, config))