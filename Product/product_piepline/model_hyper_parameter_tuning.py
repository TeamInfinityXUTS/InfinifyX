from clearml import OutputModel, Task
from clearml.automation import (
    DiscreteParameterRange,
    HyperParameterOptimizer,
    UniformIntegerParameterRange,
    UniformParameterRange,
)
from clearml.automation.optuna import OptimizerOptuna


PROJECT_NAME = "MLOps_Product_Assisted_Driving"
BASE_TASK_NAME = "GNN_Hyper_Parameter_Tuning_Base"
OPTIMIZER_TASK_NAME = "GNN_Hyper_Parameter_Tuning_Controller"
EXECUTION_QUEUE = "data_engineer"
RUN_MODE_CONTROLLER = "controller"
RUN_MODE_TRIAL = "trial"

DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = ""
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"

MAX_CONCURRENT_TASKS = 2
TOTAL_MAX_JOBS = 2
REPORT_TOP_EXPERIMENTS = 3
HPO_REPORT_PERIOD_MINUTES = 1
MIN_ITERATION_PER_JOB = 1
MAX_ITERATION_PER_JOB = 30


def train_evaluate_single_task(
    graph_cache=DEFAULT_GRAPH_CACHE,
    graph_cache_model_id=None,
    graph_cache_task_id=DEFAULT_GRAPH_CACHE_TASK_ID,
    graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
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
    from torch_geometric.nn import GCNConv, global_mean_pool

    task = Task.init(
        project_name="MLOps_Product_Assisted_Driving",
        task_name="GNN_Hyper_Parameter_Tuning_Base",
        reuse_last_task_id=False,
    )

    config = {
        "run_mode": RUN_MODE_TRIAL,
        "hidden_dim": 128,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 10,
        "batch_size": 8,
        "graph_cache": graph_cache,
        "graph_cache_model_id": graph_cache_model_id,
        "graph_cache_task_id": graph_cache_task_id,
        "graph_cache_artifact_name": graph_cache_artifact_name,
    }
    config = task.connect(config, name="General")

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
        if config.get("graph_cache_model_id"):
            input_model = InputModel(model_id=config["graph_cache_model_id"])
            return validate_file(
                input_model.get_local_copy(),
                f"ClearML model file {config['graph_cache_model_id']}",
            )

        if config.get("graph_cache_task_id"):
            return resolve_task_artifact(
                config["graph_cache_task_id"],
                config["graph_cache_artifact_name"],
            )

        cache_path = config["graph_cache"]
        if cache_path and str(cache_path).startswith("file://"):
            return validate_file(file_url_to_path(cache_path), "Graph cache")

        return validate_file(cache_path, "Graph cache")

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

    def risk_to_class(score, low_t, high_t):
        if score < low_t:
            return 0
        if score < high_t:
            return 1
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cache_path = resolve_graph_cache_path()
    print("Resolved graph cache:", graph_cache_path)

    data = torch.load(graph_cache_path, weights_only=False)
    raw_dataset = RawGraphDataset(data)
    train_size = int(0.8 * len(raw_dataset))
    train_set = Subset(raw_dataset, range(train_size))
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check graph_cache contents.")

    model = SpatioTemporalModel(
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    checkpoint_path = f"risk_model_{task.id}.pt"

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
        task.get_logger().report_scalar("metrics", "accuracy", 0.0, epoch + 1)
        print(f"Epoch {epoch + 1}/{config['epochs']} | Loss {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": config["hidden_dim"],
                    "dropout": config["dropout"],
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"],
                    "train_loss": best_loss,
                },
                checkpoint_path,
            )
            print(f"Best model saved: {checkpoint_path} loss={best_loss:.6f}")

    eval_dataset = EvalGraphDataset(data)
    test_set = Subset(
        eval_dataset,
        list(range(train_size, len(eval_dataset))),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    eval_model = SpatioTemporalModel(
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model.eval()

    train_scores = [
        eval_dataset[i][1].item()
        for i in range(train_size)
    ]
    low_t, high_t = np.percentile(train_scores, [33, 66])
    print("Thresholds:", low_t, high_t)

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

    task.get_logger().report_text(report)
    metric_iteration = int(config["epochs"]) + 1
    task.get_logger().report_scalar(
        "metrics",
        "accuracy",
        float(accuracy),
        metric_iteration,
    )
    task.get_logger().report_scalar(
        "metrics",
        "macro_precision",
        float(precision),
        metric_iteration,
    )
    task.get_logger().report_scalar(
        "metrics",
        "macro_recall",
        float(recall),
        metric_iteration,
    )
    task.get_logger().report_scalar(
        "metrics",
        "macro_f1",
        float(f1),
        metric_iteration,
    )

    print("\n===== HPO Trial Evaluation Result =====")
    print(report)
    print(f"Accuracy: {accuracy:.3f}")

    model_output = OutputModel(
        task=task,
        name="Risk_GNN_Model",
        tags=["GNN", "risk", "hpo"],
    )
    model_output.update_weights(checkpoint_path)
    model_output.set_metadata("accuracy", float(accuracy))
    model_output.set_metadata("macro_f1", float(f1))
    print("Model registered:", model_output.id)

    task_id = task.id
    task.close()
    return task_id


def run_trial_from_current_task():
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
    from torch_geometric.nn import GCNConv, global_mean_pool

    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name=PROJECT_NAME,
            task_name=BASE_TASK_NAME,
            reuse_last_task_id=False,
        )

    config = {
        "run_mode": RUN_MODE_TRIAL,
        "hidden_dim": 128,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 10,
        "batch_size": 8,
        "graph_cache": DEFAULT_GRAPH_CACHE,
        "graph_cache_model_id": None,
        "graph_cache_task_id": DEFAULT_GRAPH_CACHE_TASK_ID,
        "graph_cache_artifact_name": DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
    }
    config = task.connect(config, name="General")

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
        if config.get("graph_cache_model_id"):
            input_model = InputModel(model_id=config["graph_cache_model_id"])
            return validate_file(
                input_model.get_local_copy(),
                f"ClearML model file {config['graph_cache_model_id']}",
            )

        if config.get("graph_cache_task_id"):
            return resolve_task_artifact(
                config["graph_cache_task_id"],
                config["graph_cache_artifact_name"],
            )

        cache_path = config["graph_cache"]
        if cache_path and str(cache_path).startswith("file://"):
            return validate_file(file_url_to_path(cache_path), "Graph cache")

        return validate_file(cache_path, "Graph cache")

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

    def risk_to_class(score, low_t, high_t):
        if score < low_t:
            return 0
        if score < high_t:
            return 1
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cache_path = resolve_graph_cache_path()
    print("Resolved graph cache:", graph_cache_path)

    data = torch.load(graph_cache_path, weights_only=False)
    raw_dataset = RawGraphDataset(data)
    train_size = int(0.8 * len(raw_dataset))
    train_set = Subset(raw_dataset, range(train_size))
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check graph_cache contents.")

    model = SpatioTemporalModel(
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    checkpoint_path = f"risk_model_{task.id}.pt"

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
        task.get_logger().report_scalar("metrics", "accuracy", 0.0, epoch + 1)
        print(f"Epoch {epoch + 1}/{config['epochs']} | Loss {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": config["hidden_dim"],
                    "dropout": config["dropout"],
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"],
                    "train_loss": best_loss,
                },
                checkpoint_path,
            )

    eval_dataset = EvalGraphDataset(data)
    test_set = Subset(
        eval_dataset,
        list(range(train_size, len(eval_dataset))),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    eval_model = SpatioTemporalModel(
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
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

    metric_iteration = int(config["epochs"]) + 1
    task.get_logger().report_text(report)
    task.get_logger().report_scalar(
        "metrics",
        "accuracy",
        float(accuracy),
        metric_iteration,
    )
    task.get_logger().report_scalar(
        "metrics",
        "macro_precision",
        float(precision),
        metric_iteration,
    )
    task.get_logger().report_scalar(
        "metrics",
        "macro_recall",
        float(recall),
        metric_iteration,
    )
    task.get_logger().report_scalar(
        "metrics",
        "macro_f1",
        float(f1),
        metric_iteration,
    )

    print("\n===== HPO Trial Evaluation Result =====")
    print(report)
    print(f"Accuracy: {accuracy:.3f}")

    model_output = OutputModel(
        task=task,
        name="Risk_GNN_Model",
        tags=["GNN", "risk", "hpo"],
    )
    model_output.update_weights(checkpoint_path)
    model_output.set_metadata("accuracy", float(accuracy))
    model_output.set_metadata("macro_f1", float(f1))
    print("Model registered:", model_output.id)

    task.close()


def run_hyperparameter_optimization(base_task_id):
    optimizer_task = Task.init(
        project_name=PROJECT_NAME,
        task_name=OPTIMIZER_TASK_NAME,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    print("\n===== HPO Config =====")
    print("Execution Queue:", EXECUTION_QUEUE)
    print(
        "Make sure a ClearML agent is listening to this queue. "
        f"For example: clearml-agent daemon --queue {EXECUTION_QUEUE}"
    )
    print("Base Task ID:", base_task_id)

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            DiscreteParameterRange(
                "General/hidden_dim",
                values=[64, 128, 256],
            ),
            UniformParameterRange(
                "General/dropout",
                min_value=0.1,
                max_value=0.5,
                step_size=0.05,
            ),
            UniformParameterRange(
                "General/lr",
                min_value=0.0001,
                max_value=0.003,
                step_size=0.0001,
            ),
            UniformParameterRange(
                "General/weight_decay",
                min_value=0.00001,
                max_value=0.001,
                step_size=0.00001,
            ),
            UniformIntegerParameterRange(
                "General/epochs",
                min_value=5,
                max_value=30,
                step_size=5,
            ),
            DiscreteParameterRange(
                "General/batch_size",
                values=[8, 16, 32],
            ),
        ],
        objective_metric_title="metrics",
        objective_metric_series="accuracy",
        objective_metric_sign="max",
        optimizer_class=OptimizerOptuna,
        execution_queue=EXECUTION_QUEUE,
        max_number_of_concurrent_tasks=MAX_CONCURRENT_TASKS,
        total_max_jobs=TOTAL_MAX_JOBS,
        min_iteration_per_job=MIN_ITERATION_PER_JOB,
        max_iteration_per_job=MAX_ITERATION_PER_JOB,
    )

    optimizer.set_report_period(HPO_REPORT_PERIOD_MINUTES)

    print("\n===== HPO Started =====")
    print("Optimizer Task ID:", optimizer_task.id)

    optimizer.start()
    optimizer.wait()

    top_experiments = optimizer.get_top_experiments(top_k=REPORT_TOP_EXPERIMENTS)

    print("\n===== Top Experiments =====")
    for rank, exp in enumerate(top_experiments, start=1):
        print(f"Top {rank}: task_id={exp.id}, name={exp.name}")

    if top_experiments:
        best_task = top_experiments[0]
        print("\n===== Best Task =====")
        print("Best Task ID:", best_task.id)
        print("Best Task Name:", best_task.name)

    optimizer.stop()
    optimizer_task.close()


def run_gnn_hyper_parameter_tuning(
    graph_cache=DEFAULT_GRAPH_CACHE,
    graph_cache_model_id=None,
    graph_cache_task_id="",
    graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
):
    base_task_id = train_evaluate_single_task(
        graph_cache=graph_cache,
        graph_cache_model_id=graph_cache_model_id,
        graph_cache_task_id=graph_cache_task_id,
        graph_cache_artifact_name=graph_cache_artifact_name,
    )
    print("\nBase Task Created:", base_task_id)
    run_hyperparameter_optimization(base_task_id)
    return base_task_id


if __name__ == "__main__":
    entry_task = Task.init(
        project_name=PROJECT_NAME,
        task_name="GNN_HPO_Entry",
        reuse_last_task_id=False,
    )
    run_mode = entry_task.get_parameter("General/run_mode")

    if run_mode == RUN_MODE_TRIAL:
        run_trial_from_current_task()
    else:
        entry_task.close()
        run_gnn_hyper_parameter_tuning()
