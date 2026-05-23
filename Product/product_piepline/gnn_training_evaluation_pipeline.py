import argparse

from clearml import OutputModel, PipelineDecorator, Task

from utils.evaluation_utils import _plot_evaluation_results


PROJECT_NAME = "MLOps_Product_Assisted_Driving"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "ac6691e3ce1b4124ae670ba8c84e331d"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"


@PipelineDecorator.component(execution_queue=args.queue)
def gnn_train_step(
    graph_cache,
    graph_cache_model_id=None,
    graph_cache_task_id=None,
    graph_cache_artifact_name="graph_cache",
    hidden_dim=64,
    dropout=0.3,
    lr=1e-3,
    epochs=10,
    batch_size=8,
):
    import os
    from urllib.parse import unquote, urlparse

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from clearml import InputModel, Task
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool

    project_name = "MLOps_Product_Assisted_Driving"
    task = Task.init(project_name=project_name, task_name="GNN_Train")

    def validate_file(path, label):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    def file_url_to_path(url):
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"{parsed.netloc}{path}"
        if path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return path

    def resolve_task_artifact(task_id, artifact_name):
        source_task = Task.get_task(task_id=task_id)
        selected_name = artifact_name
        if selected_name not in source_task.artifacts:
            matches = [
                name
                for name in source_task.artifacts
                if artifact_name in name
            ]
            if matches:
                selected_name = matches[0]

        if selected_name not in source_task.artifacts:
            available = ", ".join(source_task.artifacts.keys()) or "none"
            raise KeyError(
                f"Artifact '{artifact_name}' not found in ClearML task {task_id}. "
                f"Available artifacts: {available}"
            )

        return validate_file(
            source_task.artifacts[selected_name].get_local_copy(),
            f"ClearML artifact {selected_name}",
        )

    def resolve_graph_cache_path():
        if graph_cache_model_id:
            model = InputModel(model_id=graph_cache_model_id)
            return validate_file(
                model.get_local_copy(),
                f"ClearML model file {graph_cache_model_id}",
            )

        if graph_cache_task_id:
            return resolve_task_artifact(
                graph_cache_task_id,
                graph_cache_artifact_name,
            )

        if graph_cache and str(graph_cache).startswith("file://"):
            return validate_file(file_url_to_path(graph_cache), "Graph cache")

        return validate_file(graph_cache, "Graph cache")

    class RawGraphDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    class SpatioTemporalModel(nn.Module):
        def __init__(self, in_dim=5, hidden_dim=64, dropout=0.3):
            super().__init__()
            self.gcn1 = GCNConv(in_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, graph):
            x = F.relu(self.gcn1(graph.x, graph.edge_index))
            x = self.dropout(x)
            x = F.relu(self.gcn2(x, graph.edge_index))
            x = self.dropout(x)
            return self.risk_head(global_mean_pool(x, graph.batch))

    graph_cache_path = resolve_graph_cache_path()
    print("Resolved graph cache:", graph_cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(graph_cache_path, weights_only=False)
    dataset = RawGraphDataset(data)
    train_size = int(0.8 * len(dataset))
    train_set = Subset(dataset, range(train_size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check graph_cache contents.")

    model = SpatioTemporalModel(
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    model_dir = os.path.join("models", "gnn_training_evaluation_pipeline")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "risk_gnn_model.pt")
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device).float().view(-1)

            preds = model(batch).view(-1)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        task.get_logger().report_scalar("train", "loss", float(avg_loss), epoch)
        print(f"Epoch {epoch + 1}/{epochs} | GNN loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "train_loss": best_loss,
                },
                model_path,
            )

    plot_dir = os.path.join(os.getcwd(), "clearml_plots")
    os.makedirs(plot_dir, exist_ok=True)
    loss_plot_path = os.path.join(plot_dir, "gnn_train_loss_curve.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="#1f77b4")
    ax.set_title("GNN Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(loss_plot_path)
    plt.close(fig)
    task.get_logger().report_image("train", "loss_curve", local_path=loss_plot_path, iteration=0)

    return model_path


@PipelineDecorator.component()
def gnn_evaluation_step(
    graph_cache,
    model_path,
    graph_cache_model_id=None,
    graph_cache_task_id=None,
    graph_cache_artifact_name="graph_cache",
):
    import os
    from urllib.parse import unquote, urlparse

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from clearml import InputModel, Task
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )
    from torch.utils.data import Dataset, Subset
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool

    project_name = "MLOps_Product_Assisted_Driving"
    task = Task.init(project_name=project_name, task_name="GNN_Evaluation")

    def validate_file(path, label):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    def file_url_to_path(url):
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"{parsed.netloc}{path}"
        if path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return path

    def resolve_task_artifact(task_id, artifact_name):
        source_task = Task.get_task(task_id=task_id)
        selected_name = artifact_name
        if selected_name not in source_task.artifacts:
            matches = [
                name
                for name in source_task.artifacts
                if artifact_name in name
            ]
            if matches:
                selected_name = matches[0]

        if selected_name not in source_task.artifacts:
            available = ", ".join(source_task.artifacts.keys()) or "none"
            raise KeyError(
                f"Artifact '{artifact_name}' not found in ClearML task {task_id}. "
                f"Available artifacts: {available}"
            )

        return validate_file(
            source_task.artifacts[selected_name].get_local_copy(),
            f"ClearML artifact {selected_name}",
        )

    def resolve_graph_cache_path():
        if graph_cache_model_id:
            model = InputModel(model_id=graph_cache_model_id)
            return validate_file(
                model.get_local_copy(),
                f"ClearML model file {graph_cache_model_id}",
            )

        if graph_cache_task_id:
            return resolve_task_artifact(
                graph_cache_task_id,
                graph_cache_artifact_name,
            )

        if graph_cache and str(graph_cache).startswith("file://"):
            return validate_file(file_url_to_path(graph_cache), "Graph cache")

        return validate_file(graph_cache, "Graph cache")

    class EvalGraphDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            if len(sample) == 2:
                graph, label = sample
                return graph, label
            if len(sample) == 3:
                x, edge_index, label = sample
                return Data(x=x, edge_index=edge_index), label
            raise ValueError(f"Unknown graph sample format: {len(sample)}")

    class SpatioTemporalModel(nn.Module):
        def __init__(self, in_dim=5, hidden_dim=64, dropout=0.3):
            super().__init__()
            self.gcn1 = GCNConv(in_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, graph):
            x = F.relu(self.gcn1(graph.x, graph.edge_index))
            x = self.dropout(x)
            x = F.relu(self.gcn2(x, graph.edge_index))
            x = self.dropout(x)
            return self.risk_head(global_mean_pool(x, graph.batch))

    def risk_to_class(score, low_t, high_t):
        if score < low_t:
            return 0
        if score < high_t:
            return 1
        return 2

    graph_cache_path = resolve_graph_cache_path()
    validate_file(model_path, "GNN model")
    print("Resolved graph cache:", graph_cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = torch.load(graph_cache_path, weights_only=False)
    dataset = EvalGraphDataset(raw_data)
    train_size = int(0.8 * len(dataset))
    test_set = Subset(dataset, list(range(train_size, len(dataset))))

    checkpoint = torch.load(model_path, map_location=device)
    model = SpatioTemporalModel(
        hidden_dim=checkpoint.get("hidden_dim", 64),
        dropout=checkpoint.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_scores = [dataset[i][1].item() for i in range(train_size)]
    low_t, high_t = np.percentile(train_scores, [33, 66])
    class_names = ["Low", "Medium", "High"]

    y_true = []
    y_pred = []
    y_true_raw = []
    y_pred_raw = []
    for graph, label in test_set:
        graph = graph.to(device)
        with torch.no_grad():
            pred_score = model(graph).item()
        y_true_raw.append(label.item())
        y_pred_raw.append(pred_score)
        y_true.append(risk_to_class(label.item(), low_t, high_t))
        y_pred.append(risk_to_class(pred_score, low_t, high_t))

    y_true_raw = np.array(y_true_raw, dtype=float)
    y_pred_raw = np.array(y_pred_raw, dtype=float)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        digits=3,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    class_precisions, class_recalls, class_f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    result_metrics = _plot_evaluation_results(
        y_true,
        y_pred,
        y_true_raw,
        y_pred_raw,
        ["Low", "Medium", "High"],
        low_t,
        high_t,
        task,
    )

    print(report)
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


@PipelineDecorator.component()
def register_gnn_model_step(model_path, metrics):
    from clearml import OutputModel, Task

    project_name = "MLOps_Product_Assisted_Driving"
    task = Task.current_task() or Task.init(
        project_name=project_name,
        task_name="GNN_Register_Model",
    )

    model = OutputModel(
        task=task,
        name="Risk_GNN_Model",
        tags=["GNN", "scene_graph", "risk"],
    )
    model.update_weights(model_path)
    for name, value in metrics.items():
        model.set_metadata(name, float(value))

    return {
        "model_type": "gnn",
        "model_path": model_path,
        "model_id": model.id,
        "score": float(metrics.get("macro_f1", 0.0)),
        "metrics": metrics,
    }


@PipelineDecorator.pipeline(
    name="GNN_Training_Evaluation_Pipeline",
    project=PROJECT_NAME,
)
def gnn_training_evaluation_pipeline(
    graph_cache=DEFAULT_GRAPH_CACHE,
    hidden_dim=64,
    dropout=0.3,
    lr=1e-3,
    epochs=10,
    batch_size=8,
):
    model_path = gnn_train_step(
        graph_cache,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    metrics = gnn_evaluation_step(graph_cache, model_path)
    return register_gnn_model_step(model_path, metrics)


@PipelineDecorator.pipeline(
    name="GNN_Training_Evaluation_From_ClearML_Model_Pipeline",
    project=PROJECT_NAME,
)
def gnn_training_evaluation_from_clearml_model_pipeline(
    graph_cache_model_id,
    graph_cache=DEFAULT_GRAPH_CACHE,
    hidden_dim=64,
    dropout=0.3,
    lr=1e-3,
    epochs=10,
    batch_size=8,
):
    model_path = gnn_train_step(
        graph_cache=graph_cache,
        graph_cache_model_id=graph_cache_model_id,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    metrics = gnn_evaluation_step(
        graph_cache=graph_cache,
        model_path=model_path,
        graph_cache_model_id=graph_cache_model_id,
    )
    return register_gnn_model_step(model_path, metrics)


@PipelineDecorator.pipeline(
    name="GNN_Training_Evaluation_From_ClearML_Artifact_Pipeline",
    project=PROJECT_NAME,
)
def gnn_training_evaluation_from_clearml_artifact_pipeline(
    graph_cache_task_id=DEFAULT_GRAPH_CACHE_TASK_ID,
    graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
    graph_cache=DEFAULT_GRAPH_CACHE,
    hidden_dim=64,
    dropout=0.3,
    lr=1e-3,
    epochs=10,
    batch_size=8,
):
    model_path = gnn_train_step(
        graph_cache=graph_cache,
        graph_cache_task_id=graph_cache_task_id,
        graph_cache_artifact_name=graph_cache_artifact_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    metrics = gnn_evaluation_step(
        graph_cache=graph_cache,
        model_path=model_path,
        graph_cache_task_id=graph_cache_task_id,
        graph_cache_artifact_name=graph_cache_artifact_name,
    )
    return register_gnn_model_step(model_path, metrics)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the GNN training and evaluation pipeline with custom parameters."
    )
    parser.add_argument(
        "--pipeline",
        choices=["local", "artifact", "model"],
        default="artifact",
        help="Which pipeline to run: local graph cache, artifact-based graph cache, or model-based graph cache.",
    )
    parser.add_argument(
        "--run_mode",
        choices=["local", "remote"],
        default="local",
        help="Run mode: local (in-process) or remote (enqueue to ClearML queue).",
    )
    parser.add_argument("--queue", default="Yolov8_training_v0.1", help="ClearML queue name for remote execution.")
    parser.add_argument("--graph_cache", default=DEFAULT_GRAPH_CACHE, help="Local graph cache path.")
    parser.add_argument("--graph_cache_task_id", default=DEFAULT_GRAPH_CACHE_TASK_ID, help="ClearML task id for graph cache artifact.")
    parser.add_argument("--graph_cache_artifact_name", default=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME, help="Artifact name for graph cache in ClearML task.")
    parser.add_argument("--graph_cache_model_id", default=None, help="ClearML model id for graph cache model.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size for the GNN.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for the GNN.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if args.run_mode == "local":
        PipelineDecorator.run_locally()
    else:
        PipelineDecorator.run_remotely(queue_name=args.queue)
    
    if args.pipeline == "local":
        result = gnn_training_evaluation_pipeline(
            graph_cache=args.graph_cache,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.pipeline == "model":
        result = gnn_training_evaluation_from_clearml_model_pipeline(
            graph_cache_model_id=args.graph_cache_model_id,
            graph_cache=args.graph_cache,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        result = gnn_training_evaluation_from_clearml_artifact_pipeline(
            graph_cache_task_id=args.graph_cache_task_id,
            graph_cache_artifact_name=args.graph_cache_artifact_name,
            graph_cache=args.graph_cache,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    print("Final GNN result:", result)
