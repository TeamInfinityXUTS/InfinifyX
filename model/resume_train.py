from ultralytics import YOLO
from datetime import datetime
import os


def resume_training(model_path, dataset_yaml, use_timestamp=False):
    model = YOLO(model_path)

    project_dir = "runs/YOLOv8_BDD100K"
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"resume_{timestamp}"
    else:
        name = None

    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=960,
        batch=8,
        project=project_dir if name else None,
        name=name,
        exist_ok=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        lr0=0.0005,
        lrf=0.01,
        freeze=[],
        save_period=2,
        resume=True,
        device='0',
        amp=True
    )

    if name:
        save_dir = os.path.join(project_dir, name)
    else:
        save_dir = "current training path last model: (last.pt)"

    print("\n Resume training finished!")
    print(f" Results saved in: {save_dir}")
    print(f" Best: {os.path.join(save_dir, 'weights/best.pt')}")
    print(f" Last: {os.path.join(save_dir, 'weights/last.pt')}")
