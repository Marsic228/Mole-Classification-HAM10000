from PIL import Image
from pathlib import Path
from validate_image_paths import load_metadata

def inspect_one_image(image_path):
    image = Image.open(image_path)

    result = {
        "path": str(image_path),
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height
    }
    return result

def is_rgb_image(image_info):
    checking = False
    
    if image_info["mode"] == "RGB":
        checking = True

    return checking

def has_expected_size(image_info, expected_width, expected_height):
    checking = False
    if image_info["width"] == expected_width and image_info["height"] == expected_height:
        checking = True
    return checking

def validate_image_info(image_info, expected_width, expected_height):
    result ={
        "path": image_info["path"],
        "is_rgb": is_rgb_image(image_info),
        "has_expected_size": has_expected_size(image_info, expected_width, expected_height)
    }
    return result

def inspect_image_sample(records, image_dir, limit):
    result = []
    image_dir = Path(image_dir)
    for record in records:
        if len(result) >= limit:
            break
        image_id = record["image_id"]
        image_path = image_dir / f"{image_id}.jpg"
        image_info = inspect_one_image(image_path)
        image_info["image_id"] = image_id
        result.append(image_info)
    return result

def summarize_image_sample(sample):
    total_checked = len(sample)
    count_modes = {}
    count_sizes = {}
    unique_sizes = []

    for image_info in sample:
        mode = image_info["mode"]
        if mode not in count_modes:
            count_modes[mode] = 1
        else:
            count_modes[mode] += 1
        size = f"{image_info['width']}x{image_info['height']}"
        if size not in count_sizes:
            count_sizes[size] = 1
        else:
            count_sizes[size] += 1
        if size not in unique_sizes:
            unique_sizes.append(size)

    all_rgb = count_modes.get("RGB", 0) == total_checked

    result = {
            "total_checked" : total_checked,
            "modes" : count_modes,
            "sizes" : count_sizes,
            "all_rgb" : all_rgb,
            "unique_sizes" : unique_sizes
        }
    return result


if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv", limit=10)

    sample = inspect_image_sample(records, "data/images", 5)
    summary = summarize_image_sample(sample)

    print(sample)
    print(len(sample))
    print(summary)