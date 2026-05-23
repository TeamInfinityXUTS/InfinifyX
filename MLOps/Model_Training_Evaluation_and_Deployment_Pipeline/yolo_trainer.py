from clearml import Task, Dataset
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from datetime import datetime
import yaml

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


class SE(nn.Module):
    def __init__(self, r=16):
        super().__init__()
        self.r = r
        self.built = False

        self.avg = nn.AdaptiveAvgPool2d(1)

    def _build(self, c):
        hidden = max(1, c // self.r)

        self.fc = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c),
            nn.Sigmoid()
        )

        self.built = True

    def forward(self, x):
        b, c, _, _ = x.shape

        if not self.built:
            self._build(c)

        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class CBAM(nn.Module):
    def __init__(self, r=16):
        super().__init__()
        self.r = r

        self.built = False

    def _build(self, c):
        hidden = max(1, c // self.r)

        self.mlp = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c)
        )

        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

        self.built = True

    def forward(self, x):
        b, c, h, w = x.shape

        if not self.built:
            self._build(c)

        avg = x.mean((2, 3))
        mx = x.amax((2, 3))

        channel = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * channel

        avg_p = x.mean(1, keepdim=True)
        max_p = x.max(1, keepdim=True)[0]

        spatial = torch.cat([avg_p, max_p], dim=1)
        spatial = torch.sigmoid(self.spatial(spatial))

        return x * spatial


class TransformerBlock(nn.Module):
    def __init__(self, c1, heads=4):
        super().__init__()
        self.c = c1
        self.attn = nn.MultiheadAttention(c1, heads, batch_first=True)
        self.norm = nn.LayerNorm(c1)

    def forward(self, x):
        b, c, h, w = x.shape

        x_flat = x.flatten(2).permute(0, 2, 1)

        attn, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm(attn + x_flat)

        return x.permute(0, 2, 1).reshape(b, c, h, w)


import ultralytics.nn.tasks as tasks

tasks.__dict__["CBAM"] = CBAM
tasks.__dict__["SE"] = SE
class YOLOTrainer:
    #def __init__(self, weight="runs/detect/train/weights/best.pt", yaml="Yolov8_custom_p2.yaml"):
    def __init__(self, weight="runs/detect/runs/YOLOv8_BDD100K_detect/YOLOv8_CBAM_SE3_20260421_105054/weights/best.pt"):
        

        import ultralytics.nn.tasks as tasks
        tasks.__dict__["CBAM"] = CBAM
        tasks.__dict__["SE"] = SE

        self.weight = weight
        self.yaml = yaml


        if weight:
            #self.model = YOLO(self.yaml).load(self.weight)
            self.model = YOLO(self.weight)
        else:
            self.model = YOLO(self.yaml)

    def train(self, data_yaml, name="exp", epochs=10):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = "runs/YOLOv8_BDD100K_detect"
        results = self.model.train(
            data=data_yaml,
        
            epochs=epochs,
            imgsz=960,
            batch=8,
        
            amp=True,
        

            lr0=0.00015,
            cos_lr=True,
            warmup_epochs=5,
            weight_decay=0.0005,
        

            mosaic=0.5,
            close_mosaic=15,
        
            mixup=0.05,
            copy_paste=0.2,
        
            scale=0.3,
            translate=0.1,
            fliplr=0.5,
        
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        
            multi_scale=False, 

            freeze=5,
            patience=50,
        
            device=0,

            project=project_dir,
            name=f"{name}_{ts}"
        )

        
        #results = self.model.train(
            #data=data_yaml,
            #epochs=150,
            #imgsz=960,
            #batch=8,

            #lr0=0.001,
            #lrf=0.01,
            #cos_lr=True,

            #mosaic=1.0,
            #mixup=0.1,
            #copy_paste=0.1,
            #hsv_h=0.015,
            #hsv_s=0.7,
            #hsv_v=0.4,

            #scale=0.7,
            #degrees=0.0,

            #weight_decay=0.0007,
            #device='0',
            #amp=True,
            #resume=True,
        #)

        print("Done:", results.save_dir)
        return results.save_dir

if __name__ == "__main__":

    Task.init(
        project_name="YOLOv8_Custom_Backbone",
        task_name="YOLOv8_CBAM_SE4"
    )

    #dm = DatasetManager("InfinifyX", "bdd100k")
    #data_root = dm.load()
    #yaml_path = dm.create_yaml(data_root)
    #yaml="Yolov8_custom_p2.yaml"
    yaml_path = "data/dataset.yaml"
    trainer = YOLOTrainer(weight = "runs/detect/runs/YOLOv8_BDD100K_detect/YOLOv8_CBAM_SE3_20260421_105054/weights/best.pt")
    trainer.train(yaml_path, name="YOLOv8_CBAM_SE4")
