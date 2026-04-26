from clearml import PipelineDecorator, Task
from DatasetManager import DatasetManager
from yolo_trainer import YOLOTrainer
from YOLOFeatureExtractor import YOLOFeatureExtractor
from CachedRiskDataset import CachedRiskDataset
from gnn_trainer import Trainer, SpatioTemporalModel
from CBAM import CBAM
from SE import SE
import torch
from sklearn.metrics import mean_squared_error
import os


@PipelineDecorator.component(cache=True)
def dataset_step(project, name):
    dm = DatasetManager(project, name)
    data_root = dm.load()
    yaml_path = dm.create_yaml(data_root)
    return yaml_path



@PipelineDecorator.component(execution_queue="gpu")
def yolo_train_step(yaml_path, weight):

    import ultralytics.nn.tasks as tasks
    tasks.__dict__["CBAM"] = CBAM
    tasks.__dict__["SE"] = SE

    Task.init(project_name="MLOps_Level1", task_name="YOLO_Train")

    trainer = YOLOTrainer(weight=weight)
    best_model_path = trainer.train(yaml_path, name="pipeline_yolo")

    return os.path.join(best_model_path, "weights", "best.pt")



@PipelineDecorator.component(cache=True)
def graph_build_step(yolo_weight, data_dir):

    extractor = YOLOFeatureExtractor(yolo_weight)
    dataset = CachedRiskDataset(data_dir, extractor)

    cache_path = "graph_cache.pt"
    torch.save(dataset.data, cache_path)

    print("Example graph:", dataset.data[0][0])
    print("Example label:", dataset.data[0][1])

    return cache_path


@PipelineDecorator.component(execution_queue="gpu")
def gnn_train_step(graph_cache):

    from clearml import Task
    Task.init(project_name="MLOps_Level1", task_name="GNN_Train")

    import torch
    data = torch.load(graph_cache, weights_only=False)


    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = SimpleDataset(data)

    train_size = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(train_size))

    trainer = Trainer(yolo_path=None)

    trainer.train(train_set, None, epochs=10)

    model_path = "risk_model.pt"
    torch.save(trainer.model.state_dict(), model_path)

    return model_path



from sklearn.metrics import classification_report, accuracy_score
import torch

@PipelineDecorator.component()
def evaluation_step(graph_cache, model_path):

    from clearml import Task
    Task.init(project_name="MLOps_Level1", task_name="Evaluation")

    data = torch.load(graph_cache, weights_only=False)

    from torch.utils.data import Dataset, Subset
    from torch_geometric.data import Data

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x, edge_index, label = self.data[idx]
            return Data(x=x, edge_index=edge_index), label

    dataset = SimpleDataset(data)


    train_size = int(0.8 * len(dataset))
    test_set = Subset(dataset, list(range(train_size, len(dataset))))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatioTemporalModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    def risk_to_class(score):
        if score < 0.3:
            return 0  # Low
        elif score < 0.7:
            return 1  # Medium
        else:
            return 2  # High

    y_true, y_pred = [], []

    for graph, label in test_set:

        graph = graph.to(device)

        with torch.no_grad():
            pred_score = model(graph).item()

        y_true.append(risk_to_class(label.item()))
        y_pred.append(risk_to_class(pred_score))


    report = classification_report(
        y_true,
        y_pred,
        target_names=["Low", "Medium", "High"],
        digits=3
    )

    acc = accuracy_score(y_true, y_pred)

    print("\n===== 📊 GNN Classification Report =====")
    print(report)
    print(f"Accuracy: {acc:.3f}")

    task = Task.current_task()

    task.get_logger().report_text(report)
    task.get_logger().report_scalar(
        "metrics", "accuracy", value=float(acc), iteration=0
    )

    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    task.get_logger().report_scalar("metrics", "macro_precision", float(precision), 0)
    task.get_logger().report_scalar("metrics", "macro_recall", float(recall), 0)
    task.get_logger().report_scalar("metrics", "macro_f1", float(f1), 0)

    return acc



@PipelineDecorator.component()
def register_model_step(model_path, metric):

    from clearml import OutputModel, Task

    task = Task.current_task()

    model = OutputModel(
        task=task,
        name="Risk_GNN_Model",
        tags=["GNN", "risk"]
    )

    model.update_weights(model_path)
    model.set_metadata("MSE", float(metric))

    print("Model registered")

    return model.id



@PipelineDecorator.pipeline(
    name="YOLO_GNN_Risk_Level1",
    project="MLOps_Level1"
)
def pipeline():

    yaml_path = dataset_step(
        project="InfinifyX",
        name="bdd100k"
    )

    yolo_weight = yolo_train_step(
        yaml_path=yaml_path,
        weight="yolov8n.pt"
    )

    graph_cache = graph_build_step(
        yolo_weight=yolo_weight,
        data_dir="data/test"
    )

    model = gnn_train_step(graph_cache)

    metric = evaluation_step(graph_cache, model)

    model_id = register_model_step(model, metric)

    return model_id


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline()
