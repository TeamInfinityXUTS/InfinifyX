from clearml import OutputModel, PipelineDecorator, Task


PROJECT_NAME = "MLOps_Product_Assisted_Driving"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = ""   # Set via parameter — no hardcoded fallback
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"


@PipelineDecorator.component(execution_queue="data_engineer")
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
    model_path = "risk_gnn_model.pt"

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

    return model_path


@PipelineDecorator.component(execution_queue="data_engineer")
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

    y_true = []
    y_pred = []
    for graph, label in test_set:
        graph = graph.to(device)
        with torch.no_grad():
            pred_score = model(graph).item()
        y_true.append(risk_to_class(label.item(), low_t, high_t))
        y_pred.append(risk_to_class(pred_score, low_t, high_t))

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

    task.get_logger().report_text(report)
    task.get_logger().report_scalar("metrics", "accuracy", float(accuracy), 0)
    task.get_logger().report_scalar("metrics", "macro_precision", float(precision), 0)
    task.get_logger().report_scalar("metrics", "macro_recall", float(recall), 0)
    task.get_logger().report_scalar("metrics", "macro_f1", float(f1), 0)

    print(report)
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


@PipelineDecorator.component(execution_queue="data_engineer")
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
def gnn_training_evaluation_pipeline(graph_cache=DEFAULT_GRAPH_CACHE):
    model_path = gnn_train_step(graph_cache)
    metrics = gnn_evaluation_step(graph_cache, model_path)
    return register_gnn_model_step(model_path, metrics)


@PipelineDecorator.pipeline(
    name="GNN_Training_Evaluation_From_ClearML_Model_Pipeline",
    project=PROJECT_NAME,
)
def gnn_training_evaluation_from_clearml_model_pipeline(
    graph_cache_model_id,
    graph_cache=DEFAULT_GRAPH_CACHE,
):
    model_path = gnn_train_step(
        graph_cache=graph_cache,
        graph_cache_model_id=graph_cache_model_id,
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
    graph_cache_task_id="",
    graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
    graph_cache=DEFAULT_GRAPH_CACHE,
):
    model_path = gnn_train_step(
        graph_cache=graph_cache,
        graph_cache_task_id=graph_cache_task_id,
        graph_cache_artifact_name=graph_cache_artifact_name,
    )
    metrics = gnn_evaluation_step(
        graph_cache=graph_cache,
        model_path=model_path,
        graph_cache_task_id=graph_cache_task_id,
        graph_cache_artifact_name=graph_cache_artifact_name,
    )
    return register_gnn_model_step(model_path, metrics)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    # Use local graph_cache.pt by default instead of hardcoded ClearML task ID
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    local_cache = os.path.join(project_root, 'models', 'gnn', 'graph_cache.pt')
    result = gnn_training_evaluation_pipeline(
        graph_cache=local_cache if os.path.exists(local_cache) else DEFAULT_GRAPH_CACHE,
    )
    print("Final GNN result:", result)
