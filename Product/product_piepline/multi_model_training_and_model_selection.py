import argparse
from clearml import PipelineDecorator, Task

from utils.evaluation_utils import _plot_evaluation_results


PROJECT_NAME = "MLOps_Product_Assisted_Driving"
EXECUTION_QUEUE = "Yolov8_training_v0.1"

DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "cc75418e3fd94e0f8d7c78a0e2b2f8e9"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"


def get_args():
    parser = argparse.ArgumentParser(
        description="Run multi-model GNN training, evaluation and selection with CLI overrides."
    )
    parser.add_argument(
        "--graph_cache",
        default=DEFAULT_GRAPH_CACHE,
        help="Local graph cache path.",
    )
    parser.add_argument(
        "--graph_cache_task_id",
        default=DEFAULT_GRAPH_CACHE_TASK_ID,
        help="ClearML task id for graph cache artifact.",
    )
    parser.add_argument(
        "--graph_cache_artifact_name",
        default=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
        help="Artifact name for graph cache in ClearML task.",
    )
    parser.add_argument(
        "--graph_cache_model_id",
        default=None,
        help="ClearML model id for graph cache model.",
    )
    parser.add_argument(
        "--run_mode",
        choices=["local", "remote"],
        default="local",
        help="Run mode: local or remote pipeline execution.",
    )
    parser.add_argument(
        "--queue",
        default=EXECUTION_QUEUE,
        help="ClearML queue name for remote execution.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for all GNN models.",
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default=None,
        help="State file to read pipeline parameters from.",
    )
    return parser.parse_args()

args = get_args()

import os
if args.state_file and os.path.exists(args.state_file):
    import json
    with open(args.state_file, "r") as f:
        state = json.load(f)
    if "graph_cache_path" in state:
        args.graph_cache = state["graph_cache_path"]
        print(f"[*] Read graph_cache from state file: {args.graph_cache}")

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


@PipelineDecorator.component(execution_queue=args.queue)
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
    graph_cache = graph_cache or DEFAULT_GRAPH_CACHE
    graph_cache_task_id = graph_cache_task_id or None
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
    
    def _ensure_plot_dir():
        import os

        output_dir = os.path.join(os.getcwd(), "clearml_plots")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


    def _save_confusion_matrix(cm, labels, path):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            title="GNN Evaluation Confusion Matrix",
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_title("GNN Evaluation Confusion Matrix", fontsize=14)
        ax.tick_params(axis="x", labelsize=10, rotation=45)
        ax.tick_params(axis="y", labelsize=10)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


    def _save_class_metrics_bar(precision_vals, recall_vals, f1_vals, labels, path):
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.bar(x - width, precision_vals, width, label="Precision", color="#4C72B0")
        ax.bar(x, recall_vals, width, label="Recall", color="#55A868")
        ax.bar(x + width, f1_vals, width, label="F1", color="#C44E52")
        ax.set_title("Per-class Precision / Recall / F1")
        ax.set_xlabel("Risk group")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend()
        for i, (p, r, f) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
            ax.text(i - width, p + 0.02, f"{p:.3f}", ha="center", va="bottom")
            ax.text(i, r + 0.02, f"{r:.3f}", ha="center", va="bottom")
            ax.text(i + width, f + 0.02, f"{f:.3f}", ha="center", va="bottom")
        ax.set_title("Per-class Precision / Recall / F1", fontsize=14)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


    def _save_risk_score_histogram(y_true_vals, y_pred_vals, path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.hist(y_true_vals, bins=20, alpha=0.5, label="True risk", color="#1f77b4")
        ax.hist(y_pred_vals, bins=20, alpha=0.5, label="Predicted risk", color="#55a868")
        ax.set_title("Risk Score Distribution")
        ax.set_xlabel("Risk score")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Risk Score Distribution", fontsize=14)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


    def _save_risk_calibration_curve(y_true_vals, y_pred_vals, path, n_bins=10):
        import matplotlib.pyplot as plt
        import numpy as np

        bins = np.linspace(
            min(y_pred_vals.min(), y_true_vals.min()),
            max(y_pred_vals.max(), y_true_vals.max()),
            n_bins + 1,
        )
        digitized = np.digitize(y_pred_vals, bins) - 1
        mean_pred = []
        mean_true = []
        for i in range(n_bins):
            mask = digitized == i
            if not np.any(mask):
                continue
            mean_pred.append(y_pred_vals[mask].mean())
            mean_true.append(y_true_vals[mask].mean())

        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        ax.plot(mean_pred, mean_true, marker="o", label="Calibration")
        ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], linestyle="--", color="gray", label="Ideal")
        ax.set_title("Risk Calibration Curve")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Mean true risk")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("Risk Calibration Curve", fontsize=14)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


    def _save_class_error_bars(error_dict, path):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = list(error_dict.keys())
        mae_vals = [error_dict[k]["mae"] for k in labels]
        rmse_vals = [error_dict[k]["rmse"] for k in labels]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#4C72B0")
        ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#55A868")
        ax.set_title("Risk Group MAE / RMSE")
        ax.set_xlabel("Risk group")
        ax.set_ylabel("Error")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        for i, (mae_val, rmse_val) in enumerate(zip(mae_vals, rmse_vals)):
            ax.text(i - width / 2, mae_val + 0.01, f"{mae_val:.3f}", ha="center", va="bottom")
            ax.text(i + width / 2, rmse_val + 0.01, f"{rmse_val:.3f}", ha="center", va="bottom")
        ax.set_title("Risk Group MAE / RMSE", fontsize=13)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


    def _save_precision_recall_threshold_plots(y_true_vals, y_score_vals, labels, path):
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true_vals, classes=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        for i, label_name in enumerate(labels):
            precision_curve, recall_curve, thresholds = precision_recall_curve(
                y_true_bin[:, i], y_score_vals[:, i]
            )
            if len(thresholds) == 0:
                continue
            ax.plot(thresholds, precision_curve[:-1], lw=2, label=f"{label_name} Precision")
            ax.plot(thresholds, recall_curve[:-1], lw=2, linestyle="--", label=f"{label_name} Recall")
        ax.set_title("Precision and Recall vs Threshold")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("Precision and Recall vs Threshold", fontsize=14)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


    def _save_one_vs_rest_confusion_matrices(y_true_vals, y_pred_vals, labels, path):
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix

        fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 5, 5), constrained_layout=True)
        if len(labels) == 1:
            axes = [axes]
        for i, label_name in enumerate(labels):
            y_true_bin = [1 if y == i else 0 for y in y_true_vals]
            y_pred_bin = [1 if y == i else 0 for y in y_pred_vals]
            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            ax = axes[i]
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{label_name} One-vs-Rest")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels([f"Not {label_name}", label_name], rotation=45, ha="right")
            ax.set_yticklabels([f"Not {label_name}", label_name])
            for j in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    ax.text(k, j, cm[j, k], ha="center", va="center",
                            color="white" if cm[j, k] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


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
    model_dir = os.path.join("models", "multi_model_training_and_model_selection")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{config['name']}_{task.id}.pt")

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
    y_true_raw = []
    y_pred_raw = []

    for graph, label in test_set:
        graph = graph.to(device)
        with torch.no_grad():
            pred_score = eval_model(graph).item()
        y_true.append(risk_to_class(label.item(), low_t, high_t))
        y_pred.append(risk_to_class(pred_score, low_t, high_t))
        y_true_raw.append(label.item())
        y_pred_raw.append(pred_score)

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

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        digits=3,
        zero_division=0,
    )
    accuracy = result_metrics["accuracy"]
    precision = result_metrics["macro_precision"]
    recall = result_metrics["macro_recall"]
    f1 = result_metrics["macro_f1"]

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

    def safe_score(item):
        if not isinstance(item, dict):
            return -1e9
    
        val = item.get(score_key, None)
        if val is None:
            return -1e9
    
        try:
            val = float(val)
            if not np.isfinite(val):
                return -1e9
            return val
        except:
            return -1e9

    clean_results = [
        r for r in results
        if r is not None and score_key in r
    ]

    best = max(clean_results, key=safe_score)
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
    graph_cache=DEFAULT_GRAPH_CACHE,
    graph_cache_model_id=None,
    graph_cache_task_id=DEFAULT_GRAPH_CACHE_TASK_ID,
    graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
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
    args = get_args()

    if args.epochs is not None:
        for cfg in GNN_MODEL_CONFIGS:
            cfg["epochs"] = args.epochs

    if args.run_mode == "remote":
        PipelineDecorator.run_remotely(queue=args.queue)
    else:
        PipelineDecorator.run_locally()

    result = gnn_multi_model_training_selection_pipeline(
        graph_cache=args.graph_cache,
        graph_cache_model_id=args.graph_cache_model_id,
        graph_cache_task_id=args.graph_cache_task_id,
        graph_cache_artifact_name=args.graph_cache_artifact_name,
    )

    print("Best GNN result:", result)
