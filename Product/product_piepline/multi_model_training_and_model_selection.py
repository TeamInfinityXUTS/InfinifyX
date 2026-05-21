from clearml import PipelineDecorator, Task


PROJECT_NAME = "MLOps_Product_Assisted_Driving"
EXECUTION_QUEUE = "Yolov8_training_v0.1"

DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "ac6691e3ce1b4124ae670ba8c84e331d"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"


GNN_MODEL_CONFIGS = [
    {
        "name": "GCN_Risk_Model",
        "architecture": "gcn",
        "hidden_dim": 128,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 1e-3,
        "batch_size": 8,
    },
    {
        "name": "GraphSAGE_Risk_Model",
        "architecture": "sage",
        "hidden_dim": 128,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 5e-4,
        "batch_size": 8,
    },
    {
        "name": "GAT_Risk_Model",
        "architecture": "gat",
        "hidden_dim": 128,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 5e-4,
        "batch_size": 8,
        "heads": 4,
    },
    {
        "name": "GIN_Risk_Model",
        "architecture": "gin",
        "hidden_dim": 128,
        "dropout": 0.3,
        "epochs": 10,
        "lr": 1e-4,
        "batch_size": 8,
    },
]


@PipelineDecorator.component(execution_queue="Yolov8_training_v0.1")
def train_evaluate_register_single_model(
    config,
    graph_cache=None,
    graph_cache_model_id=None,
    graph_cache_task_id=None,
    graph_cache_artifact_name=None,
):
    import os
    from urllib.parse import unquote, urlparse

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from clearml import InputModel, OutputModel, Task
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )
    from torch.utils.data import Dataset, Subset
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        GATConv,
        GCNConv,
        GINConv,
        SAGEConv,
        global_mean_pool,
    )

    task = Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name=f"Train_Evaluate_{config['name']}",
        reuse_last_task_id=False,
    )
    config = task.connect(config, name="ModelConfig")
    graph_cache = graph_cache or "graph_cache.pt"
    graph_cache_task_id = (
        graph_cache_task_id
        or "ac6691e3ce1b4124ae670ba8c84e331d"
    )
    graph_cache_artifact_name = graph_cache_artifact_name or "graph_cache"

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

    class MultiArchitectureRiskGNN(nn.Module):
        def __init__(
            self,
            architecture="gcn",
            in_dim=5,
            hidden_dim=128,
            dropout=0.3,
            heads=4,
        ):
            super().__init__()
            self.architecture = architecture
            self.dropout = nn.Dropout(dropout)

            if architecture == "gcn":
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                output_dim = hidden_dim
            elif architecture == "sage":
                self.conv1 = SAGEConv(in_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, hidden_dim)
                output_dim = hidden_dim
            elif architecture == "gat":
                self.conv1 = GATConv(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )
                self.conv2 = GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
                output_dim = hidden_dim
            elif architecture == "gin":
                self.conv1 = GINConv(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                self.conv2 = GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                output_dim = hidden_dim
            else:
                raise ValueError(f"Unsupported GNN architecture: {architecture}")

            self.risk_head = nn.Sequential(
                nn.Linear(output_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, graph):
            x = F.relu(self.conv1(graph.x, graph.edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, graph.edge_index))
            x = self.dropout(x)
            return self.risk_head(global_mean_pool(x, graph.batch))

    def risk_to_class(score, low_t, high_t):
        if score < low_t:
            return 0
        if score < high_t:
            return 1
        return 2

    graph_cache_path = resolve_graph_cache_path()
    print("Resolved graph cache:", graph_cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = torch.load(graph_cache_path, weights_only=False)

    raw_dataset = RawGraphDataset(raw_data)
    train_size = int(0.8 * len(raw_dataset))
    train_set = Subset(raw_dataset, range(train_size))
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check graph_cache contents.")

    model = MultiArchitectureRiskGNN(
        architecture=config["architecture"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        heads=config.get("heads", 4),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    model_path = f"{config['name']}_{task.id}.pt"

    for epoch in range(config["epochs"]):
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
        task.get_logger().report_scalar("train", "loss", float(avg_loss), epoch + 1)
        print(f"{config['name']} epoch {epoch + 1}/{config['epochs']} loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": config["architecture"],
                    "hidden_dim": config["hidden_dim"],
                    "dropout": config["dropout"],
                    "heads": config.get("heads", 4),
                    "lr": config["lr"],
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"],
                    "train_loss": best_loss,
                },
                model_path,
            )

    eval_dataset = EvalGraphDataset(raw_data)
    test_set = Subset(
        eval_dataset,
        list(range(train_size, len(eval_dataset))),
    )

    checkpoint = torch.load(model_path, map_location=device)
    eval_model = MultiArchitectureRiskGNN(
        architecture=checkpoint["architecture"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
        heads=checkpoint.get("heads", 4),
    ).to(device)
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model.eval()

    train_scores = [
        eval_dataset[i][1].item()
        for i in range(train_size)
    ]
    low_t, high_t = np.percentile(train_scores, [33, 66])

    y_true = []
    y_pred = []

    for graph, label in test_set:
        graph = graph.to(device)
        with torch.no_grad():
            pred_score = eval_model(graph).item()
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

    metric_iteration = int(config["epochs"])
    task.get_logger().report_text(report)
    task.get_logger().report_scalar("metrics", "accuracy", float(accuracy), metric_iteration)
    task.get_logger().report_scalar("metrics", "macro_precision", float(precision), metric_iteration)
    task.get_logger().report_scalar("metrics", "macro_recall", float(recall), metric_iteration)
    task.get_logger().report_scalar("metrics", "macro_f1", float(f1), metric_iteration)

    print(report)

    output_model = OutputModel(
        task=task,
        name=f"Risk_{config['name']}_Model",
        tags=["GNN", "multi-model", config["name"]],
    )
    output_model.update_weights(model_path)
    output_model.set_metadata("architecture", config["architecture"])
    output_model.set_metadata("accuracy", float(accuracy))
    output_model.set_metadata("macro_f1", float(f1))

    return {
        "model_type": "gnn",
        "model_name": config["name"],
        "architecture": config["architecture"],
        "model_path": model_path,
        "model_id": output_model.id,
        "score": float(f1),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


@PipelineDecorator.component()
def select_best_model(results, score_key="score"):
    from clearml import Task

    task = Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name="Best_GNN_Model_Selection",
        reuse_last_task_id=False,
    )

    print("\n===== Multi Model Results =====")
    for result in results:
        print(result)

    best = max(results, key=lambda item: float(item.get(score_key, 0.0)))
    task.get_logger().report_text(f"Best model: {best}")
    task.set_parameter("best_model_name", best["model_name"])
    task.set_parameter("best_architecture", best["architecture"])
    task.set_parameter("best_model_id", best["model_id"])
    task.set_parameter("best_score", float(best.get(score_key, 0.0)))

    return {
        "best_model": best["model_name"],
        "best_architecture": best["architecture"],
        "best_model_id": best["model_id"],
        "best_score": float(best.get(score_key, 0.0)),
        "best_accuracy": float(best.get("accuracy", 0.0)),
        "best_macro_f1": float(best.get("macro_f1", 0.0)),
        "all_results": results,
    }


@PipelineDecorator.pipeline(
    name="GNN_Multi_Model_Training_Selection_Pipeline",
    project="MLOps_Product_Assisted_Driving",
)
def gnn_multi_model_training_selection_pipeline(
    graph_cache="graph_cache.pt",
    graph_cache_model_id=None,
    graph_cache_task_id="ac6691e3ce1b4124ae670ba8c84e331d",
    graph_cache_artifact_name="graph_cache",
):
    results = []

    for config in GNN_MODEL_CONFIGS:
        result = train_evaluate_register_single_model(
            config=config,
            graph_cache=graph_cache,
            graph_cache_model_id=graph_cache_model_id,
            graph_cache_task_id=graph_cache_task_id,
            graph_cache_artifact_name=graph_cache_artifact_name,
        )
        results.append(result)

    return select_best_model(results)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    result = gnn_multi_model_training_selection_pipeline()
    print("Best GNN result:", result)
