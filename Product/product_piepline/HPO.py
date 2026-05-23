import os
from urllib.parse import unquote, urlparse

from clearml import InputModel, PipelineDecorator, Task


PROJECT_NAME = "HPO_Pipeline1"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "ac6691e3ce1b4124ae670ba8c84e331d"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"
EXECUTION_QUEUE = "HPO"


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
    import torch, numpy as np
    import torch.nn as nn, torch.nn.functional as F
    from clearml import Task
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    Task.init(project_name="HPO_Pipeline", task_name="GNN_HPO")

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
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[HPO trial {idx+1}] config={cfg} f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1; best_cfg = cfg

    print(f"[HPO] Best config: {best_cfg} f1={best_f1:.3f}")
    return {"best_config": best_cfg, "best_f1": float(best_f1)}


if __name__ == "__main__":
    Task.init(project_name="HPO_Pipeline1", task_name="GNN_HPO", reuse_last_task_id=False)
    
    result = hpo_step(
        graph_cache=DEFAULT_GRAPH_CACHE,
        graph_cache_model_id=None,
        graph_cache_task_id=DEFAULT_GRAPH_CACHE_TASK_ID,
        graph_cache_artifact_name=DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
        n_trials=2,
        epochs_per_trial=2,
    )
    print(f"HPO Result: {result}")
