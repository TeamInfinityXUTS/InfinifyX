from clearml import PipelineDecorator, Task, OutputModel
from DatasetManager import DatasetManager
from yolo_trainer import YOLOTrainer
from CBAM import CBAM
from SE import SE
from ultralytics import YOLO
import os
import torch


PROJECT_NAME = "MLOps_Product_Assisted_Driving"
DEFAULT_DATASET_PROJECT = "InfinifyX"
DEFAULT_DATASET_NAME = "bdd100k"
DEFAULT_TARGET_FOLDER = "data"
DEFAULT_WEIGHT = (
    "Model/Yolo_best.pt"
)
DEFAULT_IMAGE_SIZE = 960
DEFAULT_BATCH_SIZE = 16


def register_yolo_custom_layers():
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["CBAM"] = CBAM
    tasks.__dict__["SE"] = SE


def get_yolo_device():
    return 0 if torch.cuda.is_available() else "cpu"


def validate_file(path, label):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


@PipelineDecorator.component(cache=False)
def dataset_step(project, name, target_folder="data"):
    import os
    import uuid

    task = Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name="YOLO_Dataset_Preparation"
    )

    if os.path.exists(target_folder) and os.listdir(target_folder):
        target_folder = os.path.join(
            "clearml_data",
            f"yolo_dataset_{task.id}_{uuid.uuid4().hex[:8]}"
        )
        print("Target folder already exists and is not empty.")
        print("Using new dataset folder:", target_folder)

    dm = DatasetManager(project, name)
    data_root = dm.load(target_folder=target_folder)
    yaml_path = dm.create_yaml(data_root)

    print("Dataset yaml:", yaml_path)
    return yaml_path


@PipelineDecorator.component(cache=False)
def dataset_yaml_step(yaml_path):
    Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name="YOLO_Dataset_Yaml"
    )

    validate_file(yaml_path, "Dataset yaml")
    print("Using existing dataset yaml:", yaml_path)
    return yaml_path


@PipelineDecorator.component(execution_queue="Yolov8_training_v0.1")
def yolo_train_step(yaml_path, weight):
    register_yolo_custom_layers()
    validate_file(yaml_path, "Dataset yaml")

    Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name="YOLO_Train_v8_CBAM_SE"
    )

    weight = (
         "Model/Yolo_best.pt"
    )

    trainer = YOLOTrainer(weight=weight)
    run_dir = trainer.train(
        yaml_path,
        name="pipeline_yolo"
    )

    best_model_path = os.path.join(
        str(run_dir),
        "weights",
        "best.pt"
    )

    validate_file(best_model_path, "Best YOLO model")
    print("Best YOLO model:", best_model_path)
    return best_model_path


@PipelineDecorator.component(execution_queue="Yolov8_training_v0.1")
def yolo_evaluation_step(
    yaml_path,
    model_path,
    imgsz=960,
    batch=16
):
    register_yolo_custom_layers()
    validate_file(yaml_path, "Dataset yaml")
    validate_file(model_path, "YOLO model")

    Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name="YOLO_Evaluation"
    )

    model = YOLO(model_path)
    results = model.val(
        data=yaml_path,
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=get_yolo_device(),
        plots=True
    )

    box_metrics = results.box
    metrics = {
        "map50_95": float(getattr(box_metrics, "map", 0.0)),
        "map50": float(getattr(box_metrics, "map50", 0.0)),
        "precision": float(getattr(box_metrics, "mp", 0.0)),
        "recall": float(getattr(box_metrics, "mr", 0.0)),
    }

    task = Task.current_task()
    logger = task.get_logger()

    for metric_name, metric_value in metrics.items():
        logger.report_scalar(
            "metrics",
            metric_name,
            metric_value,
            0
        )

    class_names = getattr(results, "names", {})
    class_maps = getattr(box_metrics, "maps", None)

    if class_maps is not None:
        for class_id, class_map in enumerate(class_maps):
            class_name = class_names.get(class_id, str(class_id))
            logger.report_scalar(
                "class_map50_95",
                class_name,
                float(class_map),
                0
            )

    print("\n===== YOLO Evaluation Result =====")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics


@PipelineDecorator.component()
def register_yolo_model_step(model_path, metrics):
    validate_file(model_path, "YOLO model")

    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="MLOps_Product_Assisted_Driving",
            task_name="YOLO_Register_Model"
        )

    model = OutputModel(
        task=task,
        name="YOLO_Urban_Road_Detection_Model",
        tags=["YOLO", "BDD100K", "urban-road-detection", "CBAM", "SE"]
    )

    model.update_weights(model_path)

    for metric_name, metric_value in metrics.items():
        model.set_metadata(
            metric_name,
            float(metric_value)
        )

    print("YOLO model registered:", model.id)

    return {
        "model_type": "yolo",
        "model_path": model_path,
        "model_id": model.id,
        "score": float(metrics.get("map50_95", 0.0)),
        "metrics": metrics,
    }


@PipelineDecorator.pipeline(
    name="YOLO_Training_Evaluation_Pipeline",
    project="MLOps_Product_Assisted_Driving"
)
def yolo_training_evaluation_pipeline(
    dataset_project="InfinifyX",
    dataset_name="bdd100k",
    target_folder="data",
    weight="Model/Yolo_best.pt",
    yaml_path=None,
    imgsz=960,
    batch=16
):
    if yaml_path:
        dataset_yaml = dataset_yaml_step(yaml_path)
    else:
        dataset_yaml = dataset_step(
            project=dataset_project,
            name=dataset_name,
            target_folder=target_folder
        )

    model_path = yolo_train_step(
        yaml_path=dataset_yaml,
        weight=weight
    )

    metrics = yolo_evaluation_step(
        yaml_path=dataset_yaml,
        model_path=model_path,
        imgsz=imgsz,
        batch=batch
    )

    return register_yolo_model_step(
        model_path=model_path,
        metrics=metrics
    )


yolo_dataset_step = dataset_step
yolo_existing_yaml_step = dataset_yaml_step


if __name__ == "__main__":
    PipelineDecorator.run_locally()

    result = yolo_training_evaluation_pipeline(
        dataset_project="InfinifyX",
        dataset_name="bdd100k",
        target_folder="data",
        weight="Model/Yolo_best.pt",
        yaml_path=None,
        imgsz=960,
        batch=16
    )

    print("Final YOLO result:", result)
