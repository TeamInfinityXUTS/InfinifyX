"""
End-to-End Assisted Driving Pipeline
======================================
Chains all 6 product pipeline stages into a single ClearML pipeline:

  1. Data Processing    → extract BDD100K subset, convert JSON → YOLO format
  2. YOLO Training      → train YOLOv8-CBAM-SE, evaluate mAP
  3. Feature Engineering → build spatial graph dataset (graph_cache.pt)
  4. GNN Training       → train SpatioTemporalModel, evaluate risk classification
  5. Hyperparameter Tuning → simplified grid search over GNN hyperparameters
  6. Multi-Model Selection  → train GCN + GraphSAGE, select best architecture

Training steps are intentionally small for quick validation:
  YOLO: 2 epochs, GNN: 3 epochs, HPO: 2 trials × 2 epochs, Multi: 2 archs × 2 epochs

Run:
  python Product/product_piepline/end_to_end_pipeline.py
"""

import os
from clearml import PipelineDecorator

# ==========================================
# 0. Secure Configuration Loading
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
local_config_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'clearml.conf'))

is_ci = os.environ.get('CI', 'false').lower() == 'true'
if not is_ci:
    if os.path.exists(local_config_path):
        os.environ['CLEARML_CONFIG_FILE'] = local_config_path
        print(f"[*] Loaded ClearML config from: {local_config_path}")
    else:
        print(f"[!] Warning: clearml.conf not found at {local_config_path}")
else:
    print("[*] CI mode: using env vars for ClearML credentials.")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Processing
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.component(cache=True, execution_queue="data_engineer")
def data_preprocessing_step(dataset_path: str, subset_percentage: int) -> str:
    """Extract BDD100K subset, convert JSON→YOLO, create dataset.yaml."""
    import os, shutil, json, random

    output_dir = os.path.join(os.path.dirname(dataset_path), "bdd100k_subset_yolo")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    class_mapping = {
        "pedestrian": 0, "rider": 1, "car": 2, "truck": 3,
        "bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7,
        "traffic light": 8, "traffic sign": 9,
    }
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for name, _ in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{name}\n")

    train_json = os.path.join(dataset_path, "train", "annotations",
                              "bdd100k_labels_images_train.json")
    if not os.path.exists(train_json):
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".json"):
                    train_json = os.path.join(root, file)
                    break

    with open(train_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON list of image annotations")

    sample_size = max(1, int(len(data) * (subset_percentage / 100.0)))
    sampled = random.sample(data, sample_size)
    split_idx = int(len(sampled) * 0.8)
    train_items, val_items = sampled[:split_idx], sampled[split_idx:]
    print(f"Sampled {sample_size} → train {len(train_items)}, val {len(val_items)}")

    img_w, img_h = 1280.0, 720.0
    for split_name, items in [("train", train_items), ("val", val_items)]:
        for item in items:
            img_name = item.get("name")
            labels = item.get("labels", [])
            img_src = None
            for root, _, files in os.walk(dataset_path):
                if img_name in files:
                    img_src = os.path.join(root, img_name)
                    break
            if not img_src:
                continue
            shutil.copy2(img_src, os.path.join(output_dir, "images", split_name, img_name))
            lbl_path = os.path.join(output_dir, "labels", split_name,
                                    os.path.splitext(img_name)[0] + ".txt")
            with open(lbl_path, "w") as f_out:
                for label in labels:
                    cat = label.get("category")
                    if cat not in class_mapping:
                        continue
                    box = label.get("box2d")
                    if not box:
                        continue
                    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                    xc, yc = ((x1+x2)/2)/img_w, ((y1+y2)/2)/img_h
                    w, h = (x2-x1)/img_w, (y2-y1)/img_h
                    xc, yc, w, h = (max(0, min(1, v)) for v in (xc, yc, w, h))
                    f_out.write(f"{class_mapping[cat]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    import yaml
    yaml.safe_dump({
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": 10,
        "names": [n for n, _ in sorted(class_mapping.items(), key=lambda x: x[1])],
    }, open(yaml_path, "w"), sort_keys=False, allow_unicode=True)

    print(f"[data_preprocessing] Done → {output_dir}")
    print(f"[data_preprocessing] YAML → {yaml_path}")
    return yaml_path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — YOLO Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def yolo_train_step(yaml_path: str, init_weight: str, epochs: int,
                    imgsz: int, batch: int) -> str:
    """Train YOLOv8 with CBAM+SE, return path to best.pt."""
    import os, torch
    from ultralytics import YOLO
    import torch.nn as nn
    import ultralytics.nn.tasks as tasks

    # ── Register CBAM + SE ──
    class CBAM(nn.Module):
        def __init__(self, r=16):
            super().__init__(); self.r = r; self.built = False
        def _build(self, c):
            h = max(1, c // self.r)
            self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c))
            self.spatial = nn.Conv2d(2, 1, 7, padding=3); self.built = True
        def forward(self, x):
            b, c, *_ = x.shape
            if not self.built: self._build(c)
            avg, mx = x.mean((2,3)), x.amax((2,3))
            ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
            x = x * ch
            sp = torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], dim=1)))
            return x * sp

    class SE(nn.Module):
        def __init__(self, r=16):
            super().__init__(); self.r = r; self.built = False; self.avg = nn.AdaptiveAvgPool2d(1)
        def _build(self, c):
            h = max(1, c // self.r)
            self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c), nn.Sigmoid())
            self.built = True
        def forward(self, x):
            b, c, *_ = x.shape
            if not self.built: self._build(c)
            return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)

    tasks.__dict__["CBAM"] = CBAM
    tasks.__dict__["SE"] = SE
    print("[yolo_train] Registered CBAM + SE")

    model = YOLO(init_weight)
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.train(
        data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, project="runs/e2e_yolo", name="yolo_cbam_se",
        lr0=0.001, cos_lr=True, warmup_epochs=1, patience=5,
    )
    best_path = os.path.join(str(results.save_dir), "weights", "best.pt")
    if not os.path.isfile(best_path):
        last_path = os.path.join(str(results.save_dir), "weights", "last.pt")
        best_path = last_path
    print(f"[yolo_train] Best model → {best_path}")
    return best_path


@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def yolo_eval_step(yaml_path: str, model_path: str, imgsz: int, batch: int) -> dict:
    """Evaluate trained YOLO model, return metrics dict."""
    import os, torch
    from ultralytics import YOLO
    import torch.nn as nn
    import ultralytics.nn.tasks as tasks
    from clearml import Task

    # Re-register CBAM + SE
    class CBAM(nn.Module):
        def __init__(self, r=16):
            super().__init__(); self.r = r; self.built = False
        def _build(self, c):
            h = max(1, c // self.r)
            self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c))
            self.spatial = nn.Conv2d(2, 1, 7, padding=3); self.built = True
        def forward(self, x):
            b, c, *_ = x.shape
            if not self.built: self._build(c)
            avg, mx = x.mean((2,3)), x.amax((2,3))
            ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
            x = x * ch
            sp = torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], dim=1)))
            return x * sp
    class SE(nn.Module):
        def __init__(self, r=16):
            super().__init__(); self.r = r; self.built = False; self.avg = nn.AdaptiveAvgPool2d(1)
        def _build(self, c):
            h = max(1, c // self.r)
            self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c), nn.Sigmoid())
            self.built = True
        def forward(self, x):
            b, c, *_ = x.shape
            if not self.built: self._build(c)
            return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)
    tasks.__dict__["CBAM"] = CBAM; tasks.__dict__["SE"] = SE

    Task.init(project_name="MLOps_Product_Assisted_Driving", task_name="YOLO_Eval")
    model = YOLO(model_path)
    results = model.val(data=yaml_path, split="val", imgsz=imgsz, batch=batch,
                        device=0 if torch.cuda.is_available() else "cpu", plots=True)
    metrics = {
        "map50_95": float(getattr(results.box, "map", 0)),
        "map50":    float(getattr(results.box, "map50", 0)),
        "precision": float(getattr(results.box, "mp", 0)),
        "recall":    float(getattr(results.box, "mr", 0)),
    }
    print(f"[yolo_eval] {metrics}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature Engineering (Graph Cache)
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.component(cache=True, execution_queue="data_engineer")
def graph_build_step(yolo_weight_path: str, yolo_cache_path: str, data_dir: str) -> str:
    """Build spatial graph dataset from YOLO detections → graph_cache.pt."""
    import os, shutil
    import numpy as np, torch
    from tqdm import tqdm
    from ultralytics import YOLO
    from torch_geometric.data import Data
    import torch.nn as nn
    import ultralytics.nn.tasks as _ul

    # ── CBAM + SE ──
    class CBAM(nn.Module):
        def __init__(self, r=16):
            super().__init__(); self.r = r; self.built = False
        def _build(self, c):
            h = max(1, c // self.r)
            self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c))
            self.spatial = nn.Conv2d(2, 1, 7, padding=3); self.built = True
        def forward(self, x):
            b, c, *_ = x.shape
            if not self.built: self._build(c)
            avg, mx = x.mean((2,3)), x.amax((2,3))
            ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
            x = x * ch
            sp = torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], dim=1)))
            return x * sp
    class SE(nn.Module):
        def __init__(self, r=16):
            super().__init__(); self.r = r; self.built = False; self.avg = nn.AdaptiveAvgPool2d(1)
        def _build(self, c):
            h = max(1, c // self.r)
            self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c), nn.Sigmoid())
            self.built = True
        def forward(self, x):
            b, c, *_ = x.shape
            if not self.built: self._build(c)
            return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)
    _ul.__dict__["CBAM"] = CBAM; _ul.__dict__["SE"] = SE
    print("[graph_build] Registered CBAM + SE")

    def _ensure_cache(src, dst):
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copyfile(src, dst)
        return dst

    # ── YOLO inference ──
    model = YOLO(_ensure_cache(yolo_weight_path, yolo_cache_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = sorted(f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"[graph_build] {len(images)} images")

    CLASS_RISK = {0: 0.9, 1: 0.6, 2: 0.8, 3: 0.85, 4: 0.7, 5: 0.75, 6: 0.95, 7: 0.4, 8: 0.3}
    dataset = []
    for img_name in tqdm(images, desc="Building graphs"):
        results = model.predict(os.path.join(data_dir, img_name),
                                device=0 if torch.cuda.is_available() else "cpu", verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                dets.append({"box": box.xyxy[0].cpu().numpy(),
                             "confidence": float(box.conf), "cls": int(box.cls)})

        # Node features [cx, cy, conf, dist, cls]
        nodes = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            nodes.append([(x1+x2)/2/1280, (y1+y2)/2/720, d["confidence"],
                          1.0/((x2-x1)*(y2-y1)+1e-6), d["cls"]])
        if not nodes:
            nodes = [[0.5, 0.5, 0.0, 1.0, 0]]
        x = torch.tensor(nodes, dtype=torch.float, device=device)

        # Proximity edges
        edges = [[i,j] for i in range(len(x)) for j in range(len(x))
                 if i != j and torch.dist(x[i,:2], x[j,:2]) < 0.3]
        if not edges: edges = [[0,0]]
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        graph = Data(x=x, edge_index=edge_index)

        # Risk score
        score = 0.0
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cx, cy = (x1+x2)/2/1280, (y1+y2)/2/720
            spatial = 1 - ((cx-0.5)**2 + (cy-0.5)**2)**0.5
            size = min((x2-x1)*(y2-y1)/(1280*720), 1.0)
            score += CLASS_RISK.get(int(d["cls"]), 0.5) * d["confidence"] * (0.7*spatial + 0.3*size)
        score = score / max(len(dets), 1)
        dataset.append((graph, torch.tensor(score, dtype=torch.float32)))

    project_root = os.path.abspath(os.path.join(os.path.dirname(yolo_weight_path), "..", ".."))
    out_dir = os.path.join(project_root, "models", "gnn")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "graph_cache.pt")
    torch.save(dataset, cache_path)
    print(f"[graph_build] Saved {len(dataset)} samples → {cache_path}")
    return cache_path


@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def upload_graph_cache_step(graph_cache_path: str) -> str:
    """Upload graph_cache.pt as ClearML artifact."""
    from clearml import Task
    import os
    task = Task.current_task()
    abs_path = os.path.abspath(graph_cache_path)
    task.upload_artifact("graph_cache", abs_path,
                         metadata={"description": "Spatial graph dataset for GNN training"})
    print(f"[upload] Artifact uploaded: {abs_path}")
    return abs_path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — GNN Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def gnn_train_step(graph_cache_path: str, hidden_dim: int, dropout: float,
                   lr: float, epochs: int, batch_size: int) -> str:
    """Train SpatioTemporalModel, return model checkpoint path."""
    import os, torch
    import torch.nn as nn, torch.nn.functional as F
    from clearml import Task
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool

    Task.init(project_name="MLOps_Product_Assisted_Driving", task_name="GNN_Train")

    class RawGraphDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    class SpatioTemporalModel(nn.Module):
        def __init__(self, in_dim=5, hidden_dim=128, dropout=0.3):
            super().__init__()
            self.gcn1 = GCNConv(in_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.drop = nn.Dropout(dropout)
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        def forward(self, g):
            x = F.relu(self.gcn1(g.x, g.edge_index)); x = self.drop(x)
            x = F.relu(self.gcn2(x, g.edge_index));   x = self.drop(x)
            return self.risk_head(global_mean_pool(x, g.batch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(graph_cache_path, weights_only=False)
    ds = RawGraphDataset(data)
    train_size = int(0.8 * len(ds))
    loader = DataLoader(Subset(ds, range(train_size)), batch_size=batch_size, shuffle=True)

    model = SpatioTemporalModel(hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss, model_path = float("inf"), "risk_gnn_model.pt"

    for ep in range(epochs):
        model.train(); total = 0
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device).float().view(-1)
            loss = loss_fn(model(batch).view(-1), labels)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        avg = total / len(loader)
        print(f"[gnn_train] Epoch {ep+1}/{epochs} loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model_state_dict": model.state_dict(), "hidden_dim": hidden_dim,
                        "dropout": dropout, "train_loss": best_loss}, model_path)

    print(f"[gnn_train] Best model → {model_path}")
    return model_path


@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def gnn_eval_step(graph_cache_path: str, model_path: str) -> dict:
    """Evaluate GNN risk classification, return metrics."""
    import os, torch, numpy as np
    import torch.nn as nn, torch.nn.functional as F
    from clearml import Task
    from torch.utils.data import Dataset, Subset
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

    Task.init(project_name="MLOps_Product_Assisted_Driving", task_name="GNN_Eval")

    class EvalGraphDataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            s = self.data[idx]
            if len(s) == 2: return s[0], s[1]
            return Data(x=s[0], edge_index=s[1]), s[2]

    class SpatioTemporalModel(nn.Module):
        def __init__(self, in_dim=5, hidden_dim=128, dropout=0.3):
            super().__init__()
            self.gcn1 = GCNConv(in_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.drop = nn.Dropout(dropout)
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        def forward(self, g):
            x = F.relu(self.gcn1(g.x, g.edge_index)); x = self.drop(x)
            x = F.relu(self.gcn2(x, g.edge_index));   x = self.drop(x)
            return self.risk_head(global_mean_pool(x, g.batch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(graph_cache_path, weights_only=False)
    ds = EvalGraphDataset(raw)
    train_size = int(0.8 * len(ds))
    test_set = Subset(ds, list(range(train_size, len(ds))))

    ckpt = torch.load(model_path, map_location=device)
    model = SpatioTemporalModel(hidden_dim=ckpt.get("hidden_dim", 128),
                                dropout=ckpt.get("dropout", 0.3)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    train_scores = [ds[i][1].item() for i in range(train_size)]
    low_t, high_t = np.percentile(train_scores, [33, 66])

    y_true, y_pred = [], []
    for g, label in test_set:
        g = g.to(device)
        with torch.no_grad(): pred = model(g).item()
        y_true.append(0 if label.item() < low_t else (1 if label.item() < high_t else 2))
        y_pred.append(0 if pred < low_t else (1 if pred < high_t else 2))

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"[gnn_eval] acc={acc:.3f} f1={f1:.3f}")
    return {"accuracy": float(acc), "macro_f1": float(f1),
            "macro_precision": float(p), "macro_recall": float(r)}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Hyperparameter Tuning (simplified grid search)
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def hpo_step(graph_cache_path: str, n_trials: int = 2, epochs_per_trial: int = 2) -> dict:
    """Simplified HPO: train a few GNN configs, return best hyperparameters."""
    import torch, numpy as np
    import torch.nn as nn, torch.nn.functional as F
    from clearml import Task
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    Task.init(project_name="MLOps_Product_Assisted_Driving", task_name="GNN_HPO")

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


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Multi-Model Training & Selection
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def multi_model_step(graph_cache_path: str, epochs: int = 2) -> dict:
    """Train GCN + GraphSAGE, select best architecture."""
    import torch, numpy as np
    import torch.nn as nn, torch.nn.functional as F
    from clearml import Task, OutputModel
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    Task.init(project_name="MLOps_Product_Assisted_Driving", task_name="Multi_Model_Selection")

    class RawDS(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): return self.d[i]

    class MultiArchGNN(nn.Module):
        def __init__(self, arch="gcn", hidden_dim=128, dropout=0.3):
            super().__init__()
            Conv = GCNConv if arch == "gcn" else SAGEConv
            self.c1 = Conv(5, hidden_dim); self.c2 = Conv(hidden_dim, hidden_dim)
            self.dp = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        def forward(self, g):
            x = F.relu(self.c1(g.x, g.edge_index)); x = self.dp(x)
            x = F.relu(self.c2(x, g.edge_index));   x = self.dp(x)
            return self.head(global_mean_pool(x, g.batch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(graph_cache_path, weights_only=False)
    ds = RawDS(raw)
    train_size = int(0.8 * len(ds))
    train_loader = DataLoader(Subset(ds, range(train_size)), batch_size=8, shuffle=True)
    test_indices = list(range(train_size, len(ds)))
    train_scores = [ds[i][1].item() for i in range(train_size)]
    low_t, high_t = np.percentile(train_scores, [33, 66])

    results = []
    for arch in ("gcn", "sage"):
        model = MultiArchGNN(arch=arch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        for ep in range(epochs):
            model.train(); total = 0
            for batch, labels in train_loader:
                batch, labels = batch.to(device), labels.to(device).float().view(-1)
                loss = nn.MSELoss()(model(batch).view(-1), labels)
                opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
            print(f"[multi_model] {arch} epoch {ep+1} loss={total/len(train_loader):.6f}")

        model.eval(); y_true, y_pred = [], []
        for i in test_indices:
            g, label = ds[i]; g = g.to(device)
            with torch.no_grad(): pred = model(g).item()
            y_true.append(0 if label.item() < low_t else (1 if label.item() < high_t else 2))
            y_pred.append(0 if pred < low_t else (1 if pred < high_t else 2))
        acc = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        results.append({"architecture": arch, "accuracy": float(acc), "macro_f1": float(f1)})
        print(f"[multi_model] {arch}: acc={acc:.3f} f1={f1:.3f}")

        # Register model
        model_path = f"risk_{arch}_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "architecture": arch,
                     "hidden_dim": 128, "dropout": 0.3}, model_path)
        out_model = OutputModel(task=Task.current_task(), name=f"Risk_{arch.upper()}_Model",
                                tags=["GNN", arch])
        out_model.update_weights(model_path)

    best = max(results, key=lambda x: x["macro_f1"])
    print(f"[multi_model] Best: {best}")
    return {"best_architecture": best["architecture"], "best_accuracy": best["accuracy"],
            "best_f1": best["macro_f1"], "all_results": results}


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE STITCHING
# ═══════════════════════════════════════════════════════════════════════════
@PipelineDecorator.pipeline(
    name="End_To_End_Assisted_Driving_Pipeline",
    project="MLOps_Product_Assisted_Driving",
    version="1.0",
    default_queue="data_engineer",
)
def end_to_end_pipeline(
    dataset_path: str,
    subset_percentage: int  = 1,
    yolo_init_weight: str  = "models/yolo/Yolov8_best.pt",
    yolo_epochs: int       = 2,
    yolo_imgsz: int        = 640,
    yolo_batch: int        = 4,
    fe_data_dir: str       = "",
    gnn_hidden_dim: int    = 128,
    gnn_dropout: float     = 0.3,
    gnn_lr: float          = 5e-4,
    gnn_epochs: int        = 3,
    gnn_batch_size: int    = 8,
    hpo_trials: int        = 2,
    hpo_epochs: int        = 2,
    multi_epochs: int      = 2,
):
    """
    End-to-End Assisted Driving Pipeline
    ─────────────────────────────────────
    Step 1  data_preprocessing_step   → BDD100K subset in YOLO format + dataset.yaml
    Step 2a yolo_train_step           → Trained YOLOv8-CBAM-SE best.pt
    Step 2b yolo_eval_step            → mAP / precision / recall metrics
    Step 3a graph_build_step          → graph_cache.pt (spatial graph dataset)
    Step 3b upload_graph_cache_step   → Upload to ClearML
    Step 4a gnn_train_step            → Trained SpatioTemporalModel
    Step 4b gnn_eval_step             → Risk classification metrics
    Step 5  hpo_step                  → Best GNN hyperparameters
    Step 6  multi_model_step          → Best architecture (GCN vs GraphSAGE)
    """
    # Step 1: Data Processing
    yaml_path = data_preprocessing_step(
        dataset_path=dataset_path,
        subset_percentage=subset_percentage,
    )

    # Step 2: YOLO Training & Evaluation
    yolo_model = yolo_train_step(
        yaml_path=yaml_path,
        init_weight=yolo_init_weight,
        epochs=yolo_epochs,
        imgsz=yolo_imgsz,
        batch=yolo_batch,
    )
    yolo_metrics = yolo_eval_step(
        yaml_path=yaml_path,
        model_path=yolo_model,
        imgsz=yolo_imgsz,
        batch=yolo_batch,
    )

    # Step 3: Feature Engineering
    yolo_cache = os.path.join(os.path.dirname(yolo_init_weight), "yolo_cache_train.pt")
    if not fe_data_dir:
        fe_data_dir = os.path.join(os.path.dirname(dataset_path),
                                   "bdd100k_subset_yolo", "images", "train")
    graph_cache = graph_build_step(
        yolo_weight_path=yolo_model,
        yolo_cache_path=yolo_cache,
        data_dir=fe_data_dir,
    )
    upload_graph_cache_step(graph_cache_path=graph_cache)

    # Step 4: GNN Training & Evaluation
    gnn_model = gnn_train_step(
        graph_cache_path=graph_cache,
        hidden_dim=gnn_hidden_dim,
        dropout=gnn_dropout,
        lr=gnn_lr,
        epochs=gnn_epochs,
        batch_size=gnn_batch_size,
    )
    gnn_metrics = gnn_eval_step(
        graph_cache_path=graph_cache,
        model_path=gnn_model,
    )

    # Step 5: Hyperparameter Tuning
    hpo_result = hpo_step(
        graph_cache_path=graph_cache,
        n_trials=hpo_trials,
        epochs_per_trial=hpo_epochs,
    )

    # Step 6: Multi-Model Selection
    multi_result = multi_model_step(
        graph_cache_path=graph_cache,
        epochs=multi_epochs,
    )

    return {
        "yolo_metrics": yolo_metrics,
        "gnn_metrics":  gnn_metrics,
        "hpo_result":   hpo_result,
        "multi_result": multi_result,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from clearml import Task as _Task

    clearml_task_id = os.environ.get('CLEARML_TASK_ID')

    if clearml_task_id:
        # ── Agent / CI mode ──────────────────────────────────────────────
        # Running inside a ClearML agent (triggered by CI or ClearML UI).
        # Read parameters from the task and use the repo root as project_root.
        print(f"[*] Agent mode. Task ID: {clearml_task_id}")
        task = _Task.init(continue_last_task=clearml_task_id)
        params = task.get_parameters().get("General", {})

        # Resolve project root from the cloned repo working directory
        project_root = os.getcwd()
        dataset_path  = params.get("dataset_path",
                        os.path.join(project_root, "data_preprocessing", "datasets", "bdd100k"))
        if not os.path.isabs(dataset_path):
            abs_path = os.path.join(project_root, dataset_path)
            dataset_path = abs_path if os.path.exists(abs_path) else os.path.abspath(dataset_path)

        yolo_weight   = os.path.join(project_root, "models", "yolo", "Yolov8_best.pt")
        subset_pct    = int(params.get("subset_percentage", 1))
        yolo_ep       = int(params.get("yolo_epochs", 2))
        gnn_ep        = int(params.get("gnn_epochs", 3))
        hpo_tr        = int(params.get("hpo_trials", 2))
        multi_ep      = int(params.get("multi_epochs", 2))
    else:
        # ── Local debug mode ─────────────────────────────────────────────
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        dataset_path = os.path.join(project_root, "data_preprocessing", "datasets", "bdd100k")
        yolo_weight  = os.path.join(project_root, "models", "yolo", "Yolov8_best.pt")
        subset_pct, yolo_ep, gnn_ep, hpo_tr, multi_ep = 1, 2, 3, 2, 2

    print(f"[*] Project root  : {project_root}")
    print(f"[*] Dataset path  : {dataset_path}")
    print(f"[*] YOLO weight   : {yolo_weight}")

    # PipelineDecorator.run_locally() runs the controller + all steps
    # in the local process (subprocess per step). No agent queue needed.
    PipelineDecorator.run_locally()

    result = end_to_end_pipeline(
        dataset_path=dataset_path,
        subset_percentage=subset_pct,
        yolo_init_weight=yolo_weight,
        yolo_epochs=yolo_ep,
        yolo_imgsz=640,
        yolo_batch=4,
        gnn_hidden_dim=128,
        gnn_dropout=0.3,
        gnn_lr=5e-4,
        gnn_epochs=gnn_ep,
        gnn_batch_size=8,
        hpo_trials=hpo_tr,
        hpo_epochs=2,
        multi_epochs=multi_ep,
    )
    print("\n===== End-to-End Pipeline Complete =====")
    print(result)
