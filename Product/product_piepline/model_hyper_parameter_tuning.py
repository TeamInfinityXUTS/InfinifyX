import os
from urllib.parse import unquote, urlparse

from clearml import InputModel, PipelineDecorator, Task


PROJECT_NAME = "HPO_Pipeline"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "e42c8c37d17a402384607e9308e5a53e"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"
EXECUTION_QUEUE = "Yolov8_training_v0.1"
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = "hybrid"


DEFAULT_SEARCH_SPACE = [
    # Small model, low LR
    {"hidden_dim": 64,  "dropout": 0.1, "lr": 1e-3,  "batch_size": 32},
    {"hidden_dim": 64,  "dropout": 0.15, "lr": 5e-4,  "batch_size": 16},
    # Medium model, moderate LR
    {"hidden_dim": 128, "dropout": 0.2, "lr": 5e-4,  "batch_size": 16},
    {"hidden_dim": 128, "dropout": 0.3, "lr": 3e-4,  "batch_size": 32},
    {"hidden_dim": 128, "dropout": 0.25, "lr": 1e-3,  "batch_size": 8},
    # Large model, careful tuning
    {"hidden_dim": 256, "dropout": 0.3, "lr": 3e-4,  "batch_size": 16},
    {"hidden_dim": 256, "dropout": 0.4, "lr": 1e-4,  "batch_size": 32},
    {"hidden_dim": 256, "dropout": 0.35, "lr": 5e-5,  "batch_size": 64},
]


@PipelineDecorator.component(cache=False, execution_queue=EXECUTION_QUEUE)
def hpo_step(
    graph_cache: str = "graph_cache.pt",
    graph_cache_model_id: str = None,
    graph_cache_task_id: str = "e42c8c37d17a402384607e9308e5a53e",
    graph_cache_artifact_name: str = "graph_cache",
    n_trials: int = 3,
    max_epochs_per_trial: int = 2,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> dict:
    """
    Returns:
        dict: best_config, best_f1 (val), test_f1, and detailed metrics
    
    Data Split: Train 70% / Val 15% / Test 15%
    Objective: 3-class Classification with CrossEntropyLoss
    """
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    from clearml import Task
    from torch_geometric.loader import DataLoader  # PyG's DataLoader for graph batching
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )
    import matplotlib.pyplot as plt

    RANDOM_SEED = 42
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_METRIC = "hybrid"

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    task = Task.current_task() or Task.init(project_name="HPO_Pipeline", task_name="GNN_HPO_V2")
    logger = task.get_logger()

    logger.report_text("=" * 100)
    logger.report_text("GNN HYPERPARAMETER OPTIMIZATION (HPO) - VERSION 2: PRODUCTION-LEVEL")
    logger.report_text("="*100)
    logger.report_text(f"✓ Train/Val/Test Split: {1-val_split-test_split:.0%} / {val_split:.0%} / {test_split:.0%}")
    logger.report_text(f"✓ Objective: 3-class Classification (CrossEntropyLoss + argmax)")
    logger.report_text(f"✓ Early Stopping: {EARLY_STOPPING_METRIC.upper()} (patience={EARLY_STOPPING_PATIENCE} epochs)")
    logger.report_text(f"✓ Label Strategy: Percentile-based thresholding (33th, 66th from train set only)")
    logger.report_text(f"✓ Random Seed: {RANDOM_SEED} (reproducible)")
    logger.report_text(f"✓ Search Space: {n_trials} configs from paper-level grid (dropout scaling + batch size coupling)")
    logger.report_text("=" * 100)

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
            matches = [n for n in source_task.artifacts if artifact_name in n]
            if matches:
                selected_name = matches[0]
        if selected_name not in source_task.artifacts:
            available = ", ".join(source_task.artifacts.keys()) or "none"
            raise KeyError(f"Artifact '{artifact_name}' not found. Available: {available}")
        return validate_file(
            source_task.artifacts[selected_name].get_local_copy(),
            f"ClearML artifact {selected_name}",
        )

    def resolve_graph_cache_path():
        if graph_cache_model_id:
            model = InputModel(model_id=graph_cache_model_id)
            return validate_file(model.get_local_copy(), f"ClearML model {graph_cache_model_id}")
        if graph_cache_task_id:
            return resolve_task_artifact(graph_cache_task_id, graph_cache_artifact_name)
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

    class GNNClassifier(nn.Module):
        """GNN Risk Classifier - Direct 3-class prediction with CrossEntropyLoss."""
        def __init__(self, hidden_dim=128, dropout=0.3):
            super().__init__()
            self.gcn1 = GCNConv(5, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3),  # 3-class output
            )

        def forward(self, graph):
            x = F.relu(self.gcn1(graph.x, graph.edge_index))
            x = self.dropout(x)
            x = F.relu(self.gcn2(x, graph.edge_index))
            x = self.dropout(x)
            graph_embedding = global_mean_pool(x, graph.batch)
            logits = self.classifier(graph_embedding)
            return logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cache_path = resolve_graph_cache_path()
    logger.report_text(f"Resolved graph cache: {graph_cache_path}")

    raw_data = torch.load(graph_cache_path, weights_only=False)
    dataset = RawGraphDataset(raw_data)
    total_size = len(dataset)

    train_size = int((1 - val_split - test_split) * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    logger.report_text(f"Data Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    train_labels_raw = [dataset[i][1].item() for i in train_indices]
    train_labels_raw = np.array(train_labels_raw)
    low_threshold = np.percentile(train_labels_raw, 33)
    high_threshold = np.percentile(train_labels_raw, 66)

    def score_to_class(score):
        if score < low_threshold:
            return 0
        elif score < high_threshold:
            return 1
        else:
            return 2

    def create_labeled_dataset(indices):
        labeled = []
        for idx in indices:
            graph, raw_score = dataset[idx]
            class_label = score_to_class(raw_score)
            labeled.append((graph, class_label))
        return labeled

    train_data = create_labeled_dataset(train_indices)
    val_data = create_labeled_dataset(val_indices)
    test_data = create_labeled_dataset(test_indices)

    # Define Search Space (Paper-level HPO with dropout scaling + batch size coupling)
    local_search_space = [
        # Small model, low LR
        {"hidden_dim": 64,  "dropout": 0.1, "lr": 1e-3,  "batch_size": 32},
        {"hidden_dim": 64,  "dropout": 0.15, "lr": 5e-4,  "batch_size": 16},
        # Medium model, moderate LR
        {"hidden_dim": 128, "dropout": 0.2, "lr": 5e-4,  "batch_size": 16},
        {"hidden_dim": 128, "dropout": 0.3, "lr": 3e-4,  "batch_size": 32},
        {"hidden_dim": 128, "dropout": 0.25, "lr": 1e-3,  "batch_size": 8},
        # Large model, careful tuning
        {"hidden_dim": 256, "dropout": 0.3, "lr": 3e-4,  "batch_size": 16},
        {"hidden_dim": 256, "dropout": 0.4, "lr": 1e-4,  "batch_size": 32},
        {"hidden_dim": 256, "dropout": 0.35, "lr": 5e-5,  "batch_size": 64},
    ]
    search_space = local_search_space[:n_trials]

    logger.report_text(f"\nSearch Space ({len(search_space)} configs):")
    for i, cfg in enumerate(search_space):
        logger.report_text(f"  Config {i+1}: {cfg}")

    best_val_f1 = -1
    best_val_loss = float('inf')
    best_cfg = None
    best_model_state = None
    trial_results = []

    for trial_idx, cfg in enumerate(search_space):
        logger.report_text(f"\n{'='*100}")
        logger.report_text(f"TRIAL {trial_idx + 1}/{len(search_space)}: {cfg}")
        logger.report_text(f"{'='*100}")

        train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False)

        model = GNNClassifier(hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        loss_fn = nn.CrossEntropyLoss()

        best_val_f1_trial = -1
        best_val_loss_trial = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs_per_trial):
            model.train()
            train_loss = 0.0
            for graph, label in train_loader:
                graph = graph.to(device)
                label = label.to(device)
                logits = model(graph)
                loss = loss_fn(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_preds, val_labels_true = [], []
            val_loss = 0.0
            with torch.no_grad():
                for graph, label in val_loader:
                    graph = graph.to(device)
                    label = label.to(device)
                    logits = model(graph)
                    batch_loss = loss_fn(logits, label)
                    val_loss += batch_loss.item()
                    preds = logits.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_true.extend(label.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_labels_true, val_preds)
            _, _, val_f1, _ = precision_recall_fscore_support(
                val_labels_true, val_preds, average="macro", zero_division=0
            )

            print(f"[T{trial_idx+1}E{epoch+1:3d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            improved = False
            if EARLY_STOPPING_METRIC == "val_loss":
                improved = val_loss < best_val_loss_trial
            elif EARLY_STOPPING_METRIC == "val_f1":
                improved = val_f1 > best_val_f1_trial
            else:
                improved = (val_f1 > best_val_f1_trial) or (val_loss < best_val_loss_trial)
            
            if improved:
                best_val_f1_trial = max(best_val_f1_trial, val_f1)
                best_val_loss_trial = min(best_val_loss_trial, val_loss)
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.report_text(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                    break

        logger.report_text(f"Trial {trial_idx+1}: val_f1={best_val_f1_trial:.4f}, val_loss={best_val_loss_trial:.4f}, epochs={epoch+1}")
        
        trial_results.append({
            "trial_id": trial_idx + 1,
            "config": cfg,
            "val_f1": best_val_f1_trial,
            "val_loss": best_val_loss_trial,
            "epochs": epoch + 1,
        })

        if best_val_f1_trial > best_val_f1:
            best_val_f1 = best_val_f1_trial
            best_val_loss = best_val_loss_trial
            best_cfg = cfg
            best_model_state_final = best_model_state

    logger.report_text(f"\n{'='*100}")
    logger.report_text("FINAL EVALUATION ON TEST SET")
    logger.report_text(f"Best Config: {best_cfg}")
    logger.report_text(f"{'='*100}")

    final_model = GNNClassifier(hidden_dim=best_cfg["hidden_dim"], dropout=best_cfg["dropout"]).to(device)
    final_model.load_state_dict(best_model_state_final)
    final_model.eval()

    test_preds, test_labels_true = [], []
    with torch.no_grad():
        test_loader = DataLoader(test_data, batch_size=best_cfg["batch_size"], shuffle=False)
        for graph, label in test_loader:
            graph = graph.to(device)
            logits = final_model(graph)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels_true.extend(label.numpy())

    test_preds = np.array(test_preds)
    test_labels_true = np.array(test_labels_true)

    test_accuracy = accuracy_score(test_labels_true, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels_true, test_preds, average="macro", zero_division=0
    )
    test_class_precisions, test_class_recalls, test_class_f1s, _ = precision_recall_fscore_support(
        test_labels_true, test_preds, labels=[0, 1, 2], average=None, zero_division=0
    )

    test_report = classification_report(
        test_labels_true, test_preds, labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        digits=3, zero_division=0,
    )

    logger.report_text(test_report)
    logger.report_text(f"Test: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}")

    plot_dir = os.path.join(os.getcwd(), "clearml_plots")
    os.makedirs(plot_dir, exist_ok=True)

    cm = confusion_matrix(test_labels_true, test_preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(3), yticks=np.arange(3),
           xticklabels=["Low", "Medium", "High"],
           yticklabels=["Low", "Medium", "High"],
           ylabel="True", xlabel="Predicted", title="Confusion Matrix (Test)")
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.tight_layout()
    cm_path = os.path.join(plot_dir, "hpo_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.report_image("HPO_Results", "confusion_matrix", local_path=cm_path, iteration=0)

    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, test_class_precisions, width, label="Precision", color="#4C72B0")
    ax.bar(x, test_class_recalls, width, label="Recall", color="#55A868")
    ax.bar(x + width, test_class_f1s, width, label="F1", color="#C44E52")
    ax.set_title("Per-class Metrics (Test Set)", fontsize=14)
    ax.set_xlabel("Risk Class"), ax.set_ylabel("Score")
    ax.set_xticks(x), ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_ylim(0, 1.05), ax.legend()
    for i, (p, r, f) in enumerate(zip(test_class_precisions, test_class_recalls, test_class_f1s)):
        ax.text(i - width, p + 0.02, f"{p:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i, r + 0.02, f"{r:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width, f + 0.02, f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    metrics_path = os.path.join(plot_dir, "hpo_metrics_bar.png")
    fig.savefig(metrics_path, dpi=150)
    plt.close(fig)
    logger.report_image("HPO_Results", "metrics_bar", local_path=metrics_path, iteration=0)

    logger.report_text(f"\n{'='*100}")
    logger.report_text("HPO RESULTS SUMMARY TABLE")
    logger.report_text(f"{'='*100}")
    logger.report_text(f"{'Trial':<8} {'Hidden':<10} {'Dropout':<10} {'LR':<12} {'BS':<6} {'Val F1':<10} {'Val Loss':<12} {'Epochs':<8}")
    logger.report_text("-" * 100)
    
    sorted_results = sorted(trial_results, key=lambda x: x["val_f1"], reverse=True)
    for i, result in enumerate(sorted_results):
        cfg = result["config"]
        rank = "best" if i == 0 else f" {i+1} "
        logger.report_text(
            f"{rank:<8} {cfg['hidden_dim']:<10} {cfg['dropout']:<10.2f} {cfg['lr']:<12.0e} "
            f"{cfg['batch_size']:<6} {result['val_f1']:<10.4f} {result['val_loss']:<12.4f} {result['epochs']:<8}"
        )
    
    logger.report_text("-" * 100)
    logger.report_text(f"Best Config (by Val F1): Trial {[r['trial_id'] for r in sorted_results][0]}")
    logger.report_text(f"Best Val F1: {sorted_results[0]['val_f1']:.4f}")
    logger.report_text(f"Best Val Loss: {sorted_results[0]['val_loss']:.4f}")
    logger.report_text(f"{'='*100}")

    logger.report_text(f"\n{'='*100}")
    logger.report_text("HPO COMPLETE - VERSION 2: PRODUCTION-LEVEL")
    logger.report_text(f"{'='*100}")

    return {
        "best_config": best_cfg,
        "val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_accuracy),
    }


if __name__ == "__main__":
    Task.init(project_name="HPO_Pipeline_3", task_name="GNN_HPO", reuse_last_task_id=False)
    result = hpo_step(
        graph_cache="graph_cache.pt",
        graph_cache_model_id=None,
        graph_cache_task_id="e42c8c37d17a402384607e9308e5a53e",
        graph_cache_artifact_name="graph_cache",
        n_trials=3,
        max_epochs_per_trial=2,
        val_split=0.15,
        test_split=0.15,
    )
    print(f"\n{'='*100}")
    print("HPO RESULT:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"{'='*100}")
