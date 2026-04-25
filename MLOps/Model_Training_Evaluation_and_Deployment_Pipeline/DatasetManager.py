import os
import yaml
import torch
from clearml import Dataset


class DatasetManager:
    def __init__(self, project, name):
        self.project = project
        self.name = name

    def load(self, target_folder="data"):
        dataset = Dataset.get(
            dataset_project=self.project,
            dataset_name=self.name
        )
        path = dataset.get_mutable_local_copy(target_folder=target_folder)
        print(f"Dataset loaded: {path}")
        return path

    def create_yaml(self, data_root):
        yaml_path = os.path.join(data_root, "dataset.yaml")

        data = {
            "path": data_root,
            "train": "train/images",
            "val": "val/images",
            "nc": 10,
            "names": [
                "car","bus","truck","person","traffic light",
                "traffic sign","bike","motor","train","other"
            ]
        }

        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        return yaml_path
