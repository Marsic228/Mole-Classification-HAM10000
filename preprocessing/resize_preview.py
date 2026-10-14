from pathlib import Path
from PIL import Image
from preprocessing.preprocessing_config import load_preprocessing_config
from inspect_images import inspect_one_image

def resize_one_image(input_path, output_path, config):
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = Image.open(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_size = (config["resize_width"], config["resize_height"])
    resized_image = image.resize(target_size)
    resized_image.save(output_path)
    return output_path

if __name__ == "__main__":
    config = load_preprocessing_config("preprocessing_config.json")

    resize_one_image(
        "data/images/ISIC_0024306.jpg",
        "data/previews/ISIC_0024306_224.jpg",
        config
    )

    info = inspect_one_image("data/previews/ISIC_0024306_224.jpg")
    print(info["width"])
    print(info["height"])