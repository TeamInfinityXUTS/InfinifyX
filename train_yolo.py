from ultralytics import YOLO
from clearml import Task, Dataset
from datetime import datetime
import os

task = Task.init(
    project_name="Yolov8_training_v0.1",
    task_name="BDD100K_FULL_Training_Yolo"
)

def get_dataset():
    dataset = Dataset.get(
        dataset_project='InfinifyX',
        dataset_name='bdd100k'
    )

    local_path = dataset.get_mutable_local_copy(
        target_folder='bdd100k_full_data'
    )

    print(f"✅ Dataset path: {local_path}")
    return local_path

data_root = get_dataset()

dataset_yaml = os.path.join(data_root, "dataset.yaml")

yaml_content = f"""
path: {data_root}

train: train/images
val: val/images

nc: 10
names: ['car','bus','truck','person','traffic light','traffic sign','bike','motor','train','other']
"""

with open(dataset_yaml, "w") as f:
    f.write(yaml_content)

print("✅ dataset.yaml created!")

model = YOLO("yolov8n.pt")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_dir = "runs/YOLOv8_BDD100K"
experiment_name = f"experiment_{timestamp}"

model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=960,
    batch=8,
    project=project_dir,
    name=experiment_name,
    exist_ok=True,
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    save_period=2,
    freeze=[],
    lr0=0.0005,
    lrf=0.01,
    device='0',
    amp=True
)

save_dir = os.path.join(project_dir, experiment_name)

print("\n Training completed!")
print(f" Results saved in: {save_dir}")
print(f" Best: {save_dir}/weights/best.pt")
print(f" Last: {save_dir}/weights/last.pt")
