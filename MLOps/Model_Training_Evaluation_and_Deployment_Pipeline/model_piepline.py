from clearml import PipelineDecorator
from DatasetManager import DatasetManager
from yolo_trainer import YOLOTrainer
from YOLOFeatureExtractor import YOLOFeatureExtractor
from CachedRiskDataset import CachedRiskDataset
from gnn_trainer import Trainer
import torch
from gnn_trainer import SpatioTemporalModel
from CBAM import CBAM
from SE import SE
from clearml import Task, Dataset
from sklearn.metrics import accuracy_score

@PipelineDecorator.component(
    name="dataset_step",
    cache=True,
    execution_queue="default"
)
def dataset_step(project, name):
    dm = DatasetManager(project, name)
    data_root = dm.load()
    yaml_path = dm.create_yaml(data_root)

    return yaml_path

@PipelineDecorator.component(
    name="yolo_train_step",
    cache=False,
    execution_queue="gpu"
)
def yolo_train_step(yaml_path, weight):

    import ultralytics.nn.tasks as tasks
    tasks.__dict__["CBAM"] = CBAM
    tasks.__dict__["SE"] = SE

    Task.init(
        project_name="MLOps_Level1 Training Pipeline",
        task_name="YOLOv8_CBAM_SE_Train"
    )

    yaml_path = "data/dataset.yaml"
    weight = "Model_save/Yolo_model/best.pt"
    trainer = YOLOTrainer(weight=weight)
    best_model_path = trainer.train(yaml_path, name="pipeline_yolo")

    return best_model_path + "/weights/best.pt"

@PipelineDecorator.component(
    name="graph_build_step",
    cache=True,
    execution_queue="default"
)
def graph_build_step(yolo_weight, data_dir):

    extractor = YOLOFeatureExtractor(yolo_weight)

    dataset = CachedRiskDataset(data_dir, extractor)

    cache_path = "graph_cache.pt"
    torch.save(dataset.data, cache_path)

    return cache_path

@PipelineDecorator.component(
    name="gnn_train_step",
    cache=False,
    execution_queue="gpu"
)
def gnn_train_step(graph_cache):
    Task.init(
        project_name="MLOps_Level1 Training Pipeline",
        task_name="GNN_Risk_Model_Train"
    )
    dataset = torch.load(graph_cache)

    train_size = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(train_size))

    trainer = Trainer(yolo_path=None)

    trainer.train(train_set, None, epochs=1)

    return "risk_model.pt"


@PipelineDecorator.component(
    name="evaluation_step",
    cache=False,
    execution_queue="default"
)
def evaluation_step(graph_cache, model_path):
    Task.init(
        project_name="MLOps_Level1 Training Pipeline",
        task_name="Model_Evaluation"
    )

    dataset = torch.load(graph_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatioTemporalModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true = []
    y_pred = []

    for graphs_batch, labels in dataset:

        graphs = [g.to(device) for g in graphs_batch]
        label = labels.item()

        with torch.no_grad():
            logits = model(graphs)
            pred = logits.argmax(dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)

    print("✅ Accuracy:", acc)

    from clearml import Task
    task = Task.current_task()
    task.get_logger().report_scalar("metrics", "accuracy", value=acc, iteration=0)

    return acc

@PipelineDecorator.component(
    name="register_model_step",
    cache=False
)
def register_model_step(model_path, accuracy):

    from clearml import OutputModel, Task

    task = Task.current_task()

    model = OutputModel(
        task=task,
        name="Risk_GNN_Model",
        tags=["GNN", "risk", "production"]
    )

    model.update_weights(model_path)

    model.set_metadata({
        "accuracy": accuracy
    })

    print("✅ Model registered")

    return model.id

@PipelineDecorator.pipeline(
    name="YOLO_GNN_Risk_Level1",
    project="MLOps_Level1",
    version="1.0"
)
def pipeline():

    # Dataset
    yaml_path = dataset_step(
        project="InfinifyX",
        name="bdd100k"
    )

    # YOLO training
    yolo_weight = yolo_train_step(
        yaml_path=yaml_path,
        weight="runs/detect/train/weights/best.pt"
    )

    # Graph build
    graph_cache = graph_build_step(
        yolo_weight=yolo_weight,
        data_dir="data/test"
    )

    # GNN train
    model = gnn_train_step(graph_cache)

    # evaluation
    acc = evaluation_step(graph_cache, model)

    #register model
    model_id = register_model_step(model, acc)

    return model_id

if __name__ == "__main__":

    PipelineDecorator.run_locally()
    pipeline()
