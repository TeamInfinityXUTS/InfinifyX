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

    Loads Yolov8_best.pt via yolo_cache_train.pt, runs inference over all
    images in data_dir, constructs spatial graphs (nodes = detected objects,
    edges = spatial proximity), computes per-frame risk scores, and saves the
    resulting list of (Data, label) tuples as graph_cache.pt.

    Args:
        yolo_weight_path: Absolute path to Yolov8_best.pt
        yolo_cache_path:  Absolute path to yolo_cache_train.pt (used as local
                          YOLO model cache; copied from weight if absent)
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

    # ---- Custom attention modules (must be registered before YOLO loads .pt) ----
    # Yolov8_best.pt was trained with CBAM and SE blocks embedded in the
    # architecture. Ultralytics looks them up in nn.tasks by name at load time,
    # so they must be injected before calling YOLO().
    import torch.nn as nn
    import ultralytics.nn.tasks as _ul_tasks

    class CBAM(nn.Module):
        def __init__(self, r=16):
            super().__init__()
            self.r = r
            self.built = False

        def _build(self, c):
            hidden = max(1, c // self.r)
            self.mlp = nn.Sequential(
                nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c)
            )
            self.spatial = nn.Conv2d(2, 1, 7, padding=3)
            self.built = True

        def forward(self, x):
            b, c, h, w = x.shape
            if not self.built:
                self._build(c)
            avg = x.mean((2, 3))
            mx  = x.amax((2, 3))
            channel = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
            x = x * channel
            avg_p   = x.mean(1, keepdim=True)
            max_p   = x.max(1, keepdim=True)[0]
            spatial = torch.sigmoid(self.spatial(torch.cat([avg_p, max_p], dim=1)))
            return x * spatial

    class SE(nn.Module):
        def __init__(self, r=16):
            super().__init__()
            self.r = r
            self.built = False
            self.avg = nn.AdaptiveAvgPool2d(1)

        def _build(self, c):
            hidden = max(1, c // self.r)
            self.fc = nn.Sequential(
                nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c), nn.Sigmoid()
            )
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

    # ---- Helper: build / reuse YOLO cache file ----
    def _build_yolo_cache(src_path: str, cache_path: str) -> str:
        if not os.path.exists(cache_path):
            print(f"[*] yolo_cache_train.pt absent — copying from weight: {src_path}")
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            shutil.copyfile(src_path, cache_path)
        else:
            print(f"[*] Using existing YOLO cache: {cache_path}")
        return cache_path

    # ---- YOLOFeatureExtractor (self-contained) ----
    class YOLOFeatureExtractor:
        def __init__(self, model_path: str, cache_path: str):
            resolved = _build_yolo_cache(model_path, cache_path)
            self.model = YOLO(resolved)

        def batch_infer(self, image_paths):
            results = self.model.predict(
                image_paths,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False
            )
            all_dets = []
            for r in results:
                detections = []
                for box in r.boxes:
                    detections.append({
                        "box": box.xyxy[0].cpu().numpy(),
                        "confidence": float(box.conf),
                        "cls": int(box.cls)
                    })
                all_dets.append(detections)
            return all_dets

        def extract(self, detections):
            """Convert raw detections to [cx, cy, conf, dist, cls] node features."""
            nodes = []
            for d in detections:
                x1, y1, x2, y2 = d["box"]
                cx   = (x1 + x2) / 2 / 1280
                cy   = (y1 + y2) / 2 / 720
                conf = d["confidence"]
                area = (x2 - x1) * (y2 - y1)
                dist = 1.0 / (area + 1e-6)
                cls  = d["cls"]
                nodes.append([cx, cy, conf, dist, cls])
            if len(nodes) == 0:
                nodes = [[0.5, 0.5, 0.0, 1.0, 0]]
            return np.array(nodes, dtype=np.float32)

    # ---- Graph construction helpers (from CachedRiskDataset logic) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[graph_build_step] Device: {device}")

    def build_graph(node_features):
        x = torch.tensor(node_features, dtype=torch.float, device=device)
        edges = []
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j and torch.dist(x[i, :2], x[j, :2]) < 0.3:
                    edges.append([i, j])
        if len(edges) == 0:
            edges = [[0, 0]]
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )
        return Data(x=x, edge_index=edge_index)

    def compute_risk_score(dets):
        CLASS_RISK = {
            0: 0.9, 1: 0.6, 2: 0.8, 3: 0.85,
            4: 0.7, 5: 0.75, 6: 0.95,
            7: 0.4, 8: 0.3
        }
        score = 0.0
        for d in dets:
            cls  = int(d["cls"])
            conf = float(d["confidence"])
            x1, y1, x2, y2 = d["box"]
            cx      = (x1 + x2) / 2 / 1280
            cy      = (y1 + y2) / 2 / 720
            spatial = 1 - ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            area    = (x2 - x1) * (y2 - y1)
            size    = min(area / (1280 * 720), 1.0)
            score  += CLASS_RISK.get(cls, 0.5) * conf * (0.7 * spatial + 0.3 * size)
        return score / max(len(dets), 1)

    # ---- Main execution ----
    print(f"[graph_build_step] YOLO weight  : {yolo_weight_path}")
    print(f"[graph_build_step] YOLO cache   : {yolo_cache_path}")
    print(f"[graph_build_step] Data dir     : {data_dir}")

    extractor = YOLOFeatureExtractor(yolo_weight_path, yolo_cache_path)

    image_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"[graph_build_step] Found {len(image_files)} images.")

    dataset = []
    for img_name in tqdm(image_files, desc="Building graph dataset"):
        path = os.path.join(data_dir, img_name)
        det   = extractor.batch_infer([path])[0]
        feat  = extractor.extract(det)
        graph = build_graph(feat)
        label = compute_risk_score(det)
        dataset.append((graph, torch.tensor(label, dtype=torch.float32)))

    print(f"[graph_build_step] Built {len(dataset)} samples.")
    if dataset:
        print(f"[graph_build_step] Example graph : {dataset[0][0]}")
        print(f"[graph_build_step] Example label : {dataset[0][1]}")

    cache_path = os.path.abspath("graph_cache.pt")
    torch.save(dataset, cache_path)
    print(f"[graph_build_step] Saved graph cache → {cache_path}")

    return cache_path


# ==========================================
# 2. Upload Component
# ==========================================
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def upload_graph_cache_step(graph_cache_path: str) -> str:
    """
    Upload graph_cache.pt as a versioned ClearML artifact attached to this
    pipeline task under the MLOps_Level2 project.

    Args:
        graph_cache_path: Absolute path to graph_cache.pt

    Returns:
        The same path (pass-through for pipeline chaining)
    """
    from clearml import Task
    import os

    task = Task.current_task()
    abs_path = os.path.abspath(graph_cache_path)

    print(f"[upload_graph_cache_step] Uploading: {abs_path}")

    task.upload_artifact(
        name="graph_cache",
        artifact_object=abs_path,
        metadata={
            "description": "Pre-built spatial graph dataset (Data, risk_score) for GNN training",
            "project":     "MLOps_Level2"
        }
    )

    print("[upload_graph_cache_step] Artifact 'graph_cache' uploaded to ClearML (MLOps_Level2).")
    return abs_path


# ==========================================
# 3. Pipeline Stitching
# ==========================================
@PipelineDecorator.pipeline(
    name="Feature_Engineering_Pipeline",
    project="MLOps_Level2",
    version="1.0",
    default_queue="data_engineer"
)
def feature_engineering_pipeline(
    yolo_weight_path: str,
    yolo_cache_path: str,
    data_dir: str
):
    """
    Feature Engineering Pipeline
    ─────────────────────────────
    Step 1 — graph_build_step (cached):
        Loads Yolov8_best.pt via yolo_cache_train.pt, runs YOLO inference on
        every image in data_dir, builds a spatial proximity graph per frame,
        computes a risk score label, and persists the dataset as graph_cache.pt.

    Step 2 — upload_graph_cache_step:
        Uploads graph_cache.pt as a versioned artifact to the MLOps_Level2
        ClearML project so downstream training pipelines can consume it.
    """
    graph_cache = graph_build_step(
        yolo_weight_path=yolo_weight_path,
        yolo_cache_path=yolo_cache_path,
        data_dir=data_dir
    )

    upload_graph_cache_step(graph_cache_path=graph_cache)

    return graph_cache


# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # ---------- Configurable paths ----------
    yolo_weight = os.path.join(project_root, "models", "yolo", "Yolov8_best.pt")
    yolo_cache  = os.path.join(project_root, "models", "yolo", "yolo_cache_train.pt")

    # Adjust data_dir to the folder containing your inference images.
    # Example: data_preprocessing/datasets/bdd100k/train/images
    data_dir = os.path.join(
        project_root, "data_preprocessing", "datasets", "bdd100k", "train", "images"
    )
    # ----------------------------------------

    print(f"[*] Project root  : {project_root}")
    print(f"[*] YOLO weight   : {yolo_weight}")
    print(f"[*] YOLO cache    : {yolo_cache}")
    print(f"[*] Data directory: {data_dir}")

    # Pre-flight checks
    for label, path in [
        ("YOLO weight", yolo_weight),
        ("YOLO cache",  yolo_cache),
        ("Data dir",    data_dir),
    ]:
        status = "OK" if os.path.exists(path) else "NOT FOUND"
        print(f"    [{status}] {label}: {path}")

    # Dispatch pipeline — steps are queued to 'data_engineer' and picked up by
    # the local clearml-agent daemon already running on that queue.
    feature_engineering_pipeline(
        yolo_weight_path=yolo_weight,
        yolo_cache_path=yolo_cache,
        data_dir=data_dir
    )
