from pathlib import Path
from preprocessing.preprocessing_config import load_preprocessing_config
from preprocessing.resize_preview import resize_one_image
from dataset_summary import load_metadata_sample

def resize_image_sample(records, image_dir, output_dir, config, limit):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    result = []
    for record in records:
        if len(result) >= limit:
            break
        image_id = record["image_id"]
        input_path = image_dir / f"{image_id}.jpg"
        output_path = output_dir / f"{image_id}_{config['resize_width']}x{config['resize_height']}.jpg"
        saved_path = resize_one_image(input_path, output_path, config)
        result.append(str(saved_path))
    return result

if __name__ == "__main__":
    records = load_metadata_sample("HAM10000_metadata.csv", 3)
    config = load_preprocessing_config("preprocessing_config.json")

    saved_paths = resize_image_sample(
    records,
    "data/images",
    "data/previews/sample",
    config,
    3
    )

    print(len(saved_paths))