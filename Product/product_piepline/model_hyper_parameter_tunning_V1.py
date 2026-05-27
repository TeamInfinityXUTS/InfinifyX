import os
from urllib.parse import unquote, urlparse

from clearml import InputModel, Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
import logging
import time
import json

os.environ["MPLBACKEND"] = "Agg"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

import matplotlib
matplotlib.use('Agg', force=True)
matplotlib.rcParams['backend'] = 'Agg'
matplotlib.rcParams['interactive'] = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "HPO_Pipeline——GNN_Hyper_Parameter_Tunning"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "b6e2aee4168040729935da12e87d4b1f"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"
EXECUTION_QUEUE = "Yolov8_training_v0.1"

RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = "hybrid"


def train_evaluate_single_trial(
    graph_cache: str = "graph_cache.pt",
    graph_cache_model_id: str = None,
    graph_cache_task_id: str = "b6e2aee4168040729935da12e87d4b1f",
    graph_cache_artifact_name: str = "graph_cache",
    max_epochs_per_trial: int = 2,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> dict:
    """Base task for HyperParameterOptimizer - trains and evaluates a single model configuration."""

    # Set Random Seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    task = Task.current_task()
    if task is None:
        task = Task.init(project_name=PROJECT_NAME, task_name="GNN_HPO_Base_Task")
    
    logger_task = task.get_logger()
    
    config = {
        "hidden_dim": 128,
        "dropout": 0.3,
        "lr": 5e-4,
        "batch_size": 16,
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
    logger_task.report_text(f"Resolved graph cache: {graph_cache_path}")

    raw_data = torch.load(graph_cache_path, weights_only=False)
    dataset = RawGraphDataset(raw_data)
    total_size = len(dataset)

    train_size = int((1 - val_split - test_split) * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    logger_task.report_text(f"Data Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

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

    # Train a single model with current configuration
    logger_task.report_text(f"\nTraining with config: hidden_dim={config['hidden_dim']}, dropout={config['dropout']}, lr={config['lr']}, batch_size={config['batch_size']}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

    model = GNNClassifier(hidden_dim=config["hidden_dim"], dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1
    best_val_loss = float('inf')
    best_model_state = None
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

        logger_task.report_scalar("train", "loss", float(train_loss), epoch + 1)
        logger_task.report_scalar("metrics", "val_loss", float(val_loss), epoch + 1)
        logger_task.report_scalar("metrics", "val_f1", float(val_f1), epoch + 1)
        logger_task.report_scalar("metrics", "accuracy", float(val_accuracy), epoch + 1)

        print(f"[E{epoch+1:3d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        improved = False
        if EARLY_STOPPING_METRIC == "val_loss":
            improved = val_loss < best_val_loss
        elif EARLY_STOPPING_METRIC == "val_f1":
            improved = val_f1 > best_val_f1
        else:
            improved = (val_f1 > best_val_f1) or (val_loss < best_val_loss)
        
        if improved:
            best_val_f1 = max(best_val_f1, val_f1)
            best_val_loss = min(best_val_loss, val_loss)
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger_task.report_text(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                break

    # Test evaluation
    logger_task.report_text(f"\n{'='*100}")
    logger_task.report_text("TEST SET EVALUATION")
    logger_task.report_text(f"{'='*100}")

    final_model = GNNClassifier(hidden_dim=config["hidden_dim"], dropout=config["dropout"]).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    test_preds, test_labels_true = [], []
    with torch.no_grad():
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

    logger_task.report_text(f"Test: Accuracy={test_accuracy:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}")

    logger_task.report_text(f"\n{'='*100}")
    logger_task.report_text("TRIAL COMPLETE")
    logger_task.report_text(f"Config: hidden_dim={config['hidden_dim']}, dropout={config['dropout']}, lr={config['lr']}, batch_size={config['batch_size']}")
    logger_task.report_text(f"Val F1: {best_val_f1:.4f}, Val Loss: {best_val_loss:.4f}")
    logger_task.report_text(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    logger_task.report_text(f"{'='*100}")

    return {
        "config": config,
        "val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_accuracy),
    }


import argparse
def get_args():
    parser = argparse.ArgumentParser(
        description="Run GNN Hyperparameter Optimization (HPO) with ClearML HyperParameterOptimizer"
    )
    parser.add_argument(
        "--run_mode",
        choices=["local", "remote"],
        default="local",
        help="Run mode: local or remote (ClearML queue).",
    )
    parser.add_argument(
        "--queue",
        default=EXECUTION_QUEUE,
        help="ClearML queue name for remote execution."
    )
    parser.add_argument(
        "--graph_cache",
        default=DEFAULT_GRAPH_CACHE,
        help="Local graph cache path."
    )
    parser.add_argument(
        "--graph_cache_task_id",
        default=DEFAULT_GRAPH_CACHE_TASK_ID,
        help="ClearML task id for graph cache artifact."
    )
    parser.add_argument(
        "--graph_cache_artifact_name",
        default=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
        help="Artifact name for graph cache in ClearML task."
    )
    parser.add_argument(
        "--graph_cache_model_id",
        default=None,
        help="ClearML model id for graph cache model."
    )
    parser.add_argument(
        "--max_epochs_per_trial",
        type=int,
        default=2,
        help="Maximum epochs per trial."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation set fraction."
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Test set fraction."
    )
    parser.add_argument(
        "--max_concurrent_tasks",
        type=int,
        default=2,
        help="Max concurrent HPO tasks."
    )
    parser.add_argument(
        "--total_max_jobs",
        type=int,
        default=2,
        help="Total max HPO jobs."
    )
    parser.add_argument(
        "--time_limit_minutes",
        type=int,
        default=20,
        help="Time limit for HPO in minutes."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Initialize the HPO task
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name='HPO: GNN Hyperparameter Optimization',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    
    # Connect parameters
    hpo_args = {
        'max_epochs_per_trial': args.max_epochs_per_trial,
        'time_limit_minutes': args.time_limit_minutes,
        'total_max_jobs': args.total_max_jobs,
        'max_concurrent_tasks': args.max_concurrent_tasks,
        'graph_cache': args.graph_cache,
        'graph_cache_task_id': args.graph_cache_task_id,
        'graph_cache_artifact_name': args.graph_cache_artifact_name,
        'graph_cache_model_id': args.graph_cache_model_id,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'run_mode': args.run_mode,
        'queue': args.queue,
    }
    hpo_args = task.connect(hpo_args)
    logger.info(f"Connected parameters: {hpo_args}")
    
    # Execute remotely if requested
    if args.run_mode == "remote":
        logger.info(f"Executing remotely on queue: {args.queue}")
        task.execute_remotely(queue_name=args.queue)
    
    print("\n===== GNN Hyperparameter Optimization =====")
    print(f"Run Mode: {args.run_mode}")
    print(f"Queue: {args.queue}")
    print(f"Max Concurrent Tasks: {args.max_concurrent_tasks}")
    print(f"Total Max Jobs: {args.total_max_jobs}")
    
    # Step 1: Create base task by initializing it with training
    print("\n[1/2] Creating and initializing base task with metrics...")
    base_task_id = task.id
    
    result = train_evaluate_single_trial(
        graph_cache=args.graph_cache,
        graph_cache_model_id=args.graph_cache_model_id,
        graph_cache_task_id=args.graph_cache_task_id,
        graph_cache_artifact_name=args.graph_cache_artifact_name,
        max_epochs_per_trial=args.max_epochs_per_trial,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    print(f"Base task initialized. Val F1: {result['val_f1']:.4f}, Test F1: {result['test_f1']:.4f}")
    
    # Step 2: Launch HPO
    print("\n[2/2] Launching HyperParameterOptimizer...")
    
    hpo_task = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            UniformIntegerParameterRange(
                "General/hidden_dim",
                min_value=64,
                max_value=256,
                step_size=64,
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
            UniformIntegerParameterRange(
                "General/batch_size",
                min_value=8,
                max_value=32,
                step_size=8,
            ),
        ],
        objective_metric_title="metrics",
        objective_metric_series="accuracy",
        objective_metric_sign="max",
        execution_queue=args.queue,
        max_number_of_concurrent_tasks=args.max_concurrent_tasks,
        total_max_jobs=args.total_max_jobs,
        pool_period_min=1.0,
        save_top_k_tasks_only=2,
    )

    print("\n===== HPO Started =====")
    print(f"Base Task ID: {base_task_id}")
    print(f"Max Epochs per Trial: {args.max_epochs_per_trial}")
    print(f"Time Limit: {args.time_limit_minutes} minutes")
    
    hpo_task.start()
    
    # Wait for optimization to complete
    logger.info(f"Waiting for optimization to complete (time limit: {args.time_limit_minutes} minutes)...")
    time.sleep(args.time_limit_minutes * 60)
    
    # Get the top performing experiments
    try:
        top_exp = hpo_task.get_top_experiments(top_k=1)
        print("\n===== Top 1 Experiments =====")
        
        if top_exp:
            for rank, exp in enumerate(top_exp, start=1):
                try:
                    params = exp.get_parameters()
                    metrics = exp.get_last_scalar_metrics()
                    
                    accuracy = None
                    if metrics and 'metrics' in metrics and 'accuracy' in metrics['metrics']:
                        accuracy = metrics['metrics']['accuracy'].get('last_value', None)
                    
                    print(f"Top {rank}: task_id={exp.id}, name={exp.name}")
                    if params:
                        print(f"  - hidden_dim: {params.get('General/hidden_dim', 'N/A')}")
                        print(f"  - dropout: {params.get('General/dropout', 'N/A')}")
                        print(f"  - lr: {params.get('General/lr', 'N/A')}")
                        print(f"  - batch_size: {params.get('General/batch_size', 'N/A')}")
                    if accuracy:
                        print(f"  - accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"Top {rank}: task_id={exp.id}, name={exp.name} (failed to get details: {e})")
        else:
            print("No experiments completed yet.")
    except Exception as e:
        logger.error(f"Failed to get top experiments: {e}")
    
    hpo_task.stop()
    logger.info("Optimizer stopped")
    
    task.close()
    
    print("\n===== HPO Complete =====")
