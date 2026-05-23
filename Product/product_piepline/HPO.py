import os
from urllib.parse import unquote, urlparse

from clearml import InputModel, PipelineDecorator, Task


PROJECT_NAME = "HPO_Pipeline"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "ac6691e3ce1b4124ae670ba8c84e331d"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"
EXECUTION_QUEUE = "Yolov8_training_v0.1"


@PipelineDecorator.component(cache=False, execution_queue=EXECUTION_QUEUE)
def hpo_step(
    graph_cache: str = "graph_cache.pt",
    graph_cache_model_id: str = None,
    graph_cache_task_id: str = "ac6691e3ce1b4124ae670ba8c84e331d",
    graph_cache_artifact_name: str = "graph_cache",
    n_trials: int = 2,
    epochs_per_trial: int = 2,
) -> dict:
    """Simplified HPO: train a few GNN configs, return best hyperparameters."""
    import os
    import torch, numpy as np
    import torch.nn as nn, torch.nn.functional as F
    from clearml import Task
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )
    import matplotlib.pyplot as plt

    task = Task.current_task() or Task.init(project_name="HPO_Pipeline", task_name="GNN_HPO")
    logger = task.get_logger()

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

    class RawDS(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): return self.d[i]

    class STM(nn.Module):
        def __init__(self, hd=128, dp=0.3):
            super().__init__()
            self.g1 = GCNConv(5, hd); self.g2 = GCNConv(hd, hd); self.dp = nn.Dropout(dp)
            self.head = nn.Sequential(nn.Linear(hd, 64), nn.ReLU(), nn.Dropout(dp), nn.Linear(64, 1))
        def forward(self, g):
            x = F.relu(self.g1(g.x, g.edge_index)); x = self.dp(x)
            x = F.relu(self.g2(x, g.edge_index));   x = self.dp(x)
            return self.head(global_mean_pool(x, g.batch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cache_path = resolve_graph_cache_path()
    print(f"Resolved graph cache: {graph_cache_path}")
    raw = torch.load(graph_cache_path, weights_only=False)
    ds = RawDS(raw)
    train_size = int(0.8 * len(ds))
    test_indices = list(range(train_size, len(ds)))
    train_scores = [ds[i][1].item() for i in range(train_size)]
    low_t, high_t = np.percentile(train_scores, [33, 66])

    search_space = [
        {"hidden_dim": 64,  "dropout": 0.2, "lr": 1e-3},
        {"hidden_dim": 128, "dropout": 0.3, "lr": 5e-4},
        {"hidden_dim": 128, "dropout": 0.4, "lr": 1e-3},
    ][:n_trials]

    best_cfg, best_f1 = None, 0
    best_accuracy = 0
    best_y_true, best_y_pred = [], []
    best_report = ""
    best_class_precisions, best_class_recalls, best_class_f1s = [], [], []
    for idx, cfg in enumerate(search_space):
        loader = DataLoader(Subset(ds, range(train_size)), batch_size=8, shuffle=True)
        model = STM(hd=cfg["hidden_dim"], dp=cfg["dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        for ep in range(epochs_per_trial):
            model.train(); total = 0
            for batch, labels in loader:
                batch, labels = batch.to(device), labels.to(device).float().view(-1)
                loss = nn.MSELoss()(model(batch).view(-1), labels)
                opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
            print(f"[HPO trial {idx+1}] epoch {ep+1} loss={total/len(loader):.6f}")

        # Quick eval
        model.eval(); y_true, y_pred = [], []
        for i in test_indices:
            g, label = ds[i]; g = g.to(device)
            with torch.no_grad(): pred = model(g).item()
            y_true.append(0 if label.item() < low_t else (1 if label.item() < high_t else 2))
            y_pred.append(0 if pred < low_t else (1 if pred < high_t else 2))
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        class_precisions, class_recalls, class_f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
        )
        
        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            target_names=["Low", "Medium", "High"],
            digits=3, zero_division=0,
        )
        
        print(f"[HPO trial {idx+1}] config={cfg} f1={f1:.3f}")
        print(f"[HPO trial {idx+1}] accuracy={accuracy:.3f}")
        
        # Log metrics to ClearML
        logger.report_scalar(f"trial_{idx+1}/metrics", "accuracy", float(accuracy), 0)
        logger.report_scalar(f"trial_{idx+1}/metrics", "precision", float(precision), 0)
        logger.report_scalar(f"trial_{idx+1}/metrics", "recall", float(recall), 0)
        logger.report_scalar(f"trial_{idx+1}/metrics", "f1", float(f1), 0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = cfg
            best_accuracy = accuracy
            best_y_true = y_true
            best_y_pred = y_pred
            best_report = report
            best_class_precisions = class_precisions
            best_class_recalls = class_recalls
            best_class_f1s = class_f1s

    # Generate evaluation plots for best trial
    plot_dir = os.path.join(os.getcwd(), "clearml_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(best_y_true, best_y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
        ylabel="True label",
        xlabel="Predicted label",
        title="GNN HPO Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = os.path.join(plot_dir, "hpo_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.report_image("HPO_Results", "confusion_matrix", local_path=cm_path, iteration=0)
    
    # Per-class Metrics Bar Chart
    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, best_class_precisions, width, label="Precision", color="#4C72B0")
    ax.bar(x, best_class_recalls, width, label="Recall", color="#55A868")
    ax.bar(x + width, best_class_f1s, width, label="F1", color="#C44E52")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.set_xlabel("Risk group")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (p, r, f) in enumerate(zip(best_class_precisions, best_class_recalls, best_class_f1s)):
        ax.text(i - width, p + 0.02, f"{p:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i, r + 0.02, f"{r:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width, f + 0.02, f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    metrics_path = os.path.join(plot_dir, "hpo_metrics_bar.png")
    fig.savefig(metrics_path, dpi=150)
    plt.close(fig)
    logger.report_image("HPO_Results", "metrics_bar_chart", local_path=metrics_path, iteration=0)
    
    # Report classification report
    logger.report_text(best_report)
    logger.report_text(f"Best HPO Trial: Config = {best_cfg}, F1 = {best_f1:.3f}")

    print(f"[HPO] Best config: {best_cfg} f1={best_f1:.3f}")
    return {"best_config": best_cfg, "best_f1": float(best_f1), "best_accuracy": float(best_accuracy)}


if __name__ == "__main__":
    # Initialize ClearML Task to avoid AttributeError with PipelineDecorator
    Task.init(project_name=PROJECT_NAME, task_name="GNN_HPO_Main", reuse_last_task_id=False)
    
    result = hpo_step(
        graph_cache=DEFAULT_GRAPH_CACHE,
        graph_cache_model_id=None,
        graph_cache_task_id=DEFAULT_GRAPH_CACHE_TASK_ID,
        graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
        n_trials=2,
        epochs_per_trial=2,
    )
    print(f"HPO Result: {result}")
