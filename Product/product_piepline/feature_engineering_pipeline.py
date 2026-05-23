"""
Feature Engineering Pipeline
=============================
Integrated copy of MLOps/FeatureEngineering/feature_engineering_pipeline.py,
adapted for the Product pipeline directory.

Responsibilities:
  1. graph_build_step (cached):
       Load Yolov8_best.pt (with CBAM + SE injection), run YOLO inference on
       every image in data_dir, build spatial proximity graphs per frame,
       compute risk-score labels, and persist the dataset as graph_cache.pt
       under models/gnn/.
  2. upload_graph_cache_step:
       Upload graph_cache.pt as a versioned ClearML artifact.

ClearML project: MLOps_Product_Assisted_Driving
Execution queue: data_engineer
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
    print("[*] CI mode: using environment variables for ClearML credentials.")


# ==========================================
# 1. Graph Build Component
# ==========================================
@PipelineDecorator.component(cache=True, execution_queue="data_engineer")
def graph_build_step(yolo_weight_path: str, yolo_cache_path: str, data_dir: str) -> str:
    """
    Feature extraction step.

    Loads Yolov8_best.pt (via yolo_cache_train.pt), runs inference over all
    images in data_dir, constructs spatial proximity graphs (nodes = detected
    objects, edges = proximity < 0.3), computes per-frame risk scores, and
    saves the resulting list[(Data, risk_score)] as models/gnn/graph_cache.pt.

    Args:
        yolo_weight_path: Absolute path to Yolov8_best.pt
        yolo_cache_path:  Absolute path to yolo_cache_train.pt
        data_dir:         Directory containing .jpg / .jpeg / .png images

    Returns:
        Absolute path to the saved graph_cache.pt
    """
    import os
    import shutil
    import numpy as np
    import torch
    from tqdm import tqdm
    from ultralytics import YOLO
    from torch_geometric.data import Data

    # ── Custom attention modules (must be registered before YOLO loads .pt) ──
    # Yolov8_best.pt was trained with CBAM and SE blocks embedded in the
    # architecture. Ultralytics resolves them by name in nn.tasks at load time.
    import torch.nn as nn
    import ultralytics.nn.tasks as _ul_tasks

    class CBAM(nn.Module):
        def __init__(self, r=16):
            super().__init__()
            self.r = r
            self.built = False

        def _build(self, c):
            hidden = max(1, c // self.r)
            self.mlp     = nn.Sequential(nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c))
            self.spatial = nn.Conv2d(2, 1, 7, padding=3)
            self.built   = True

        def forward(self, x):
            b, c, h, w = x.shape
            if not self.built:
                self._build(c)
            avg     = x.mean((2, 3))
            mx      = x.amax((2, 3))
            channel = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
            x       = x * channel
            spatial = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True),
                                                             x.max(1, keepdim=True)[0]], dim=1)))
            return x * spatial

    class SE(nn.Module):
        def __init__(self, r=16):
            super().__init__()
            self.r     = r
            self.built = False
            self.avg   = nn.AdaptiveAvgPool2d(1)

        def _build(self, c):
            hidden  = max(1, c // self.r)
            self.fc = nn.Sequential(nn.Linear(c, hidden), nn.ReLU(),
                                    nn.Linear(hidden, c), nn.Sigmoid())
            self.built = True

        def forward(self, x):
            b, c, _, _ = x.shape
            if not self.built:
                self._build(c)
            y = self.avg(x).view(b, c)
            return x * self.fc(y).view(b, c, 1, 1)

    _ul_tasks.__dict__["CBAM"] = CBAM
    _ul_tasks.__dict__["SE"]   = SE
    print("[graph_build_step] Registered CBAM and SE into ultralytics.nn.tasks")

    # ── YOLO cache helper ────────────────────────────────────────────────────
    def _build_yolo_cache(src_path: str, cache_path: str) -> str:
        if not os.path.exists(cache_path):
            print(f"[*] yolo_cache_train.pt absent — copying from: {src_path}")
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            shutil.copyfile(src_path, cache_path)
        else:
            print(f"[*] Using existing YOLO cache: {cache_path}")
        return cache_path

    # ── YOLOFeatureExtractor (self-contained for agent isolation) ────────────
    class YOLOFeatureExtractor:
        def __init__(self, model_path: str, cache_path: str):
            self.model = YOLO(_build_yolo_cache(model_path, cache_path))

        def batch_infer(self, image_paths):
            results  = self.model.predict(
                image_paths,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )
            all_dets = []
            for r in results:
                all_dets.append([
                    {"box": box.xyxy[0].cpu().numpy(),
                     "confidence": float(box.conf),
                     "cls": int(box.cls)}
                    for box in r.boxes
                ])
            return all_dets

        def extract(self, detections):
            """Convert raw detections → [cx, cy, conf, dist, cls] node features."""
            nodes = []
            for d in detections:
                x1, y1, x2, y2 = d["box"]
                cx   = (x1 + x2) / 2 / 1280
                cy   = (y1 + y2) / 2 / 720
                conf = d["confidence"]
                area = (x2 - x1) * (y2 - y1)
                dist = 1.0 / (area + 1e-6)
                nodes.append([cx, cy, conf, dist, d["cls"]])
            return np.array(nodes if nodes else [[0.5, 0.5, 0.0, 1.0, 0]],
                            dtype=np.float32)

    # ── Graph construction ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[graph_build_step] Device: {device}")

    def build_graph(node_features):
        x     = torch.tensor(node_features, dtype=torch.float, device=device)
        edges = [[i, j] for i in range(len(x)) for j in range(len(x))
                 if i != j and torch.dist(x[i, :2], x[j, :2]) < 0.3]
        if not edges:
            edges = [[0, 0]]
        return Data(
            x=x,
            edge_index=torch.tensor(edges, dtype=torch.long, device=device).t().contiguous(),
        )

    def compute_risk_score(dets):
        CLASS_RISK = {0: 0.9, 1: 0.6, 2: 0.8, 3: 0.85,
                      4: 0.7, 5: 0.75, 6: 0.95, 7: 0.4, 8: 0.3}
        score = 0.0
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cx      = (x1 + x2) / 2 / 1280
            cy      = (y1 + y2) / 2 / 720
            spatial = 1 - ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            area    = (x2 - x1) * (y2 - y1)
            size    = min(area / (1280 * 720), 1.0)
            score  += (CLASS_RISK.get(int(d["cls"]), 0.5)
                       * float(d["confidence"])
                       * (0.7 * spatial + 0.3 * size))
        return score / max(len(dets), 1)

    # ── Main execution ───────────────────────────────────────────────────────
    print(f"[graph_build_step] YOLO weight  : {yolo_weight_path}")
    print(f"[graph_build_step] YOLO cache   : {yolo_cache_path}")
    print(f"[graph_build_step] Data dir     : {data_dir}")

    extractor   = YOLOFeatureExtractor(yolo_weight_path, yolo_cache_path)
    image_files = sorted(f for f in os.listdir(data_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"[graph_build_step] Found {len(image_files)} images.")

    dataset = []
    for img_name in tqdm(image_files, desc="Building graph dataset"):
        path  = os.path.join(data_dir, img_name)
        det   = extractor.batch_infer([path])[0]
        feat  = extractor.extract(det)
        graph = build_graph(feat)
        label = compute_risk_score(det)
        dataset.append((graph, torch.tensor(label, dtype=torch.float32)))

    print(f"[graph_build_step] Built {len(dataset)} samples.")

    # Derive project root from yolo_weight_path (models/yolo/… → ../../)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(yolo_weight_path), '..', '..')
    )
    output_dir   = os.path.join(project_root, "models", "gnn")
    os.makedirs(output_dir, exist_ok=True)
    cache_path   = os.path.join(output_dir, "graph_cache.pt")

    torch.save(dataset, cache_path)
    print(f"[graph_build_step] Saved graph cache → {cache_path}")
    return cache_path


# ==========================================
# 2. Upload Component
# ==========================================
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def upload_graph_cache_step(graph_cache_path: str) -> str:
    """Upload graph_cache.pt as a versioned ClearML artifact."""
    from clearml import Task
    import os

    task     = Task.current_task()
    abs_path = os.path.abspath(graph_cache_path)
    print(f"[upload_graph_cache_step] Uploading: {abs_path}")

    task.upload_artifact(
        name="graph_cache",
        artifact_object=abs_path,
        metadata={
            "description": "Pre-built spatial graph dataset (Data, risk_score) for GNN training",
            "project":     "MLOps_Product_Assisted_Driving",
        },
    )
    print("[upload_graph_cache_step] Artifact 'graph_cache' uploaded.")
    return abs_path


# ==========================================
# 3. Pipeline Stitching
# ==========================================
@PipelineDecorator.pipeline(
    name="Feature_Engineering_Pipeline",
    project="MLOps_Product_Assisted_Driving",
    version="1.0",
    default_queue="data_engineer",
)
def feature_engineering_pipeline(
    yolo_weight_path: str,
    yolo_cache_path: str,
    data_dir: str,
):
    """
    Feature Engineering Pipeline
    ─────────────────────────────
    Step 1 — graph_build_step (cached):
        YOLO inference → spatial proximity graph → risk label → graph_cache.pt
    Step 2 — upload_graph_cache_step:
        Uploads graph_cache.pt to ClearML (MLOps_Product_Assisted_Driving).
    """
    graph_cache = graph_build_step(
        yolo_weight_path=yolo_weight_path,
        yolo_cache_path=yolo_cache_path,
        data_dir=data_dir,
    )
    upload_graph_cache_step(graph_cache_path=graph_cache)
    return graph_cache


# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    import argparse
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_weight", default=os.path.join(project_root, "models", "yolo", "Yolov8_best.pt"))
    parser.add_argument("--yolo_cache", default=os.path.join(project_root, "models", "yolo", "yolo_cache_train.pt"))
    parser.add_argument("--data_dir", default=os.path.join(
        project_root, "data_preprocessing", "datasets",
        "bdd100k_subset_yolo", "images", "train"
    ))
    args = parser.parse_args()

    print(f"[*] Project root  : {project_root}")
    print(f"[*] YOLO weight   : {args.yolo_weight}")
    print(f"[*] YOLO cache    : {args.yolo_cache}")
    print(f"[*] Data directory: {args.data_dir}")

    for label, path in [("YOLO weight", args.yolo_weight),
                         ("YOLO cache",  args.yolo_cache),
                         ("Data dir",    args.data_dir)]:
        print(f"    [{'OK' if os.path.exists(path) else 'NOT FOUND'}] {label}: {path}")

    # Run pipeline controller locally — individual steps still dispatch to
    # the 'data_engineer' queue and are picked up by the local ClearML agent.
    PipelineDecorator.run_locally()

    feature_engineering_pipeline(
        yolo_weight_path=args.yolo_weight,
        yolo_cache_path=args.yolo_cache,
        data_dir=args.data_dir,
    )
