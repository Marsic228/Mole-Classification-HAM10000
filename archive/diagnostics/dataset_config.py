# dataset_config.py
# Goal: store and validate dataset configuration for Mole Classification
# You need: class, __init__, validation, to_dict, from_dict, __repr__

import json



class DatasetConfig:
    def __init__(self, metadata_path, image_dirs, num_classes, image_size, split_ratios):
        self.metadata_path = metadata_path
        self.image_dirs = image_dirs
        self.num_classes = num_classes
        self.image_size = image_size
        self.split_ratios = split_ratios
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be greater than 0")
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2 or self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("Image size must contain two positive values")
        if not isinstance(self.split_ratios, (tuple, list)) or len(self.split_ratios) !=3 or abs(sum(self.split_ratios) - 1.0) > 0.01:
            raise ValueError("Split ratios must contain three values that sum to 1.0")
        if not self.image_dirs:
            raise ValueError("Image dirs cannot be empty")
    

    def to_dict(self):
        return{
            "metadata_path" : self.metadata_path,
            "image_dirs" : self.image_dirs,
            "num_classes" : self.num_classes,
            "image_size" : list(self.image_size),
            "split_ratios" : list(self.split_ratios)
        }
        

    @classmethod
    def from_dict(cls, data):
        return cls(
            metadata_path=data.get("metadata_path"),
            image_dirs=data.get("image_dirs"),
            num_classes=data.get("num_classes"),
            image_size=tuple(data.get("image_size")),
            split_ratios=data.get("split_ratios")
        )
        

    def __repr__(self):
        return f"DatasetConfig(metadata: {self.metadata_path}, image_dirs: {self.image_dirs}, num_classes: {self.num_classes}, image_size: {self.image_size}, split_ratios: {self.split_ratios})"
        


def save_config(config, path):
    with open(path, "w") as file:
         json.dump(config.to_dict(), file)



def load_config(path):
    with open(path, "r") as file:
        result =  json.load(file)
        return DatasetConfig.from_dict(result)



if __name__ == "__main__":
    config = DatasetConfig(
        metadata_path = "data/HAM10000_metadata.csv",
        image_dirs = ["data/images_part_1", "data/images_part_2"],
        num_classes = 7,
        image_size = (372, 372),
        split_ratios = [0.7, 0.15, 0.15]
    )
    print(config)
    path = "dataset_config.json"
    save_config(config, path)
    loaded = load_config(path)
    print(loaded)
   