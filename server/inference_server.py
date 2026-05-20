"""
InfinifyX Inference Server
==========================
FastAPI server that:
  - Loads Yolov8_best.pt (with CBAM / SE attention blocks) + risk_model.pt (GNN)
  - Exposes a WebSocket endpoint at ws://localhost:8000/ws
  - Receives raw JPEG frames from the browser, runs YOLO → GNN inference,
    and returns JSON: { boxes: [...], risk: {label, class, score} }
  - Serves the browser client (server/static/index.html) at GET /

Run:
    cd <project_root>
    python server/inference_server.py
    # or with auto-reload during development:
    uvicorn server.inference_server:app --reload --port 8000
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SERVER_DIR   = Path(__file__).parent
PROJECT_ROOT = SERVER_DIR.parent
YOLO_WEIGHT  = PROJECT_ROOT / "models" / "yolo" / "Yolov8_best.pt"
GNN_WEIGHT   = PROJECT_ROOT / "models" / "gnn"  / "risk_model.pt"
STATIC_DIR   = SERVER_DIR / "static"

# ──────────────────────────────────────────────────────────────────────────────
# Risk classification thresholds  (percentile-based approximation)
# Low < 0.33  ≤  Medium  < 0.66  ≤  High
# ──────────────────────────────────────────────────────────────────────────────
LOW_THRESHOLD  = 0.33
HIGH_THRESHOLD = 0.66
RISK_LABELS    = {0: "Low", 1: "Medium", 2: "High"}


# ──────────────────────────────────────────────────────────────────────────────
# Custom attention modules  (identical to CBAM.py / SE.py used during training)
# Must be injected into ultralytics.nn.tasks before loading Yolov8_best.pt
# ──────────────────────────────────────────────────────────────────────────────

class CBAM(nn.Module):
    """Channel + Spatial Attention Block (lazy-build on first forward pass)."""

    def __init__(self, r: int = 16):
        super().__init__()
        self.r     = r
        self.built = False

    def _build(self, c: int):
        hidden      = max(1, c // self.r)
        self.mlp    = nn.Sequential(nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.built  = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if not self.built:
            self._build(c)
        avg     = x.mean((2, 3))
        mx      = x.amax((2, 3))
        channel = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x       = x * channel
        avg_p   = x.mean(1, keepdim=True)
        max_p   = x.max(1, keepdim=True)[0]
        return x * torch.sigmoid(self.spatial(torch.cat([avg_p, max_p], dim=1)))


class SE(nn.Module):
    """Squeeze-and-Excitation Block (lazy-build on first forward pass)."""

    def __init__(self, r: int = 16):
        super().__init__()
        self.r     = r
        self.built = False
        self.avg   = nn.AdaptiveAvgPool2d(1)

    def _build(self, c: int):
        hidden   = max(1, c // self.r)
        self.fc  = nn.Sequential(
            nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c), nn.Sigmoid()
        )
        self.built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        if not self.built:
            self._build(c)
        return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# GNN model  (must match SpatioTemporalModel in gnn_trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalModel(nn.Module):
    """Two-layer GCN → global mean pool → scalar risk score."""

    def __init__(self, in_dim: int = 5, hidden: int = 64):
        super().__init__()
        self.gcn1      = GCNConv(in_dim, hidden)
        self.gcn2      = GCNConv(hidden, hidden)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, g: Data) -> torch.Tensor:
        x     = F.relu(self.gcn1(g.x, g.edge_index))
        x     = F.relu(self.gcn2(x, g.edge_index))
        batch = (g.batch if (hasattr(g, "batch") and g.batch is not None)
                 else torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
        x     = global_mean_pool(x, batch)
        return self.risk_head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_yolo(weight_path: Path):
    """Inject CBAM / SE into ultralytics namespace, then load YOLO weights."""
    import ultralytics.nn.tasks as ul_tasks
    ul_tasks.__dict__["CBAM"] = CBAM
    ul_tasks.__dict__["SE"]   = SE
    print(f"[server] Registered CBAM + SE into ultralytics.nn.tasks")

    from ultralytics import YOLO
    print(f"[server] Loading YOLO  ← {weight_path}")
    model = YOLO(str(weight_path))
    print(f"[server] YOLO ready  (classes: {list(model.names.values())})")
    return model


def load_gnn(weight_path: Path):
    """Load SpatioTemporalModel checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpatioTemporalModel().to(device)
    ckpt   = torch.load(str(weight_path), map_location=device, weights_only=False)
    # checkpoint may be {"model_state_dict": ...} or a raw state_dict
    state  = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[server] GNN ready    ← {weight_path}  (device: {device})")
    return model, device


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def bytes_to_frame(data: bytes):
    """Decode raw JPEG bytes → BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_yolo(yolo, frame: np.ndarray) -> list:
    """Run YOLO on a BGR frame, return list of detection dicts."""
    results = yolo.predict(
        frame,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "box":        [x1, y1, x2, y2],
                "cls":        int(box.cls),
                "cls_name":   yolo.names[int(box.cls)],
                "confidence": float(box.conf),
            })
    return detections


def build_graph(detections: list, device: torch.device, img_w=1280, img_h=720) -> Data:
    """
    Convert YOLO detections to a torch_geometric Data graph.
    Node features: [cx, cy, conf, dist, cls]  (5-dim, matches training)
    Edges: spatial proximity (normalised distance < 0.3)
    """
    nodes = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cx   = (x1 + x2) / 2 / img_w
        cy   = (y1 + y2) / 2 / img_h
        conf = d["confidence"]
        area = (x2 - x1) * (y2 - y1)
        dist = 1.0 / (area + 1e-6)
        nodes.append([cx, cy, conf, dist, float(d["cls"])])

    if not nodes:
        nodes = [[0.5, 0.5, 0.0, 1.0, 0.0]]   # sentinel for empty frame

    x = torch.tensor(nodes, dtype=torch.float, device=device)
    edges = [
        [i, j]
        for i in range(len(x))
        for j in range(len(x))
        if i != j and torch.dist(x[i, :2], x[j, :2]).item() < 0.3
    ]
    if not edges:
        edges = [[0, 0]]

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def classify_risk(score: float) -> dict:
    """Map continuous risk score to Low / Medium / High."""
    cls = 0 if score < LOW_THRESHOLD else (1 if score < HIGH_THRESHOLD else 2)
    return {"label": RISK_LABELS[cls], "class": cls, "score": round(score, 4)}


def infer_frame(yolo, gnn, device, frame: np.ndarray) -> dict:
    """Full inference pipeline for one frame."""
    detections = run_yolo(yolo, frame)
    graph      = build_graph(detections, device)
    with torch.no_grad():
        score = float(gnn(graph).item())
    return {
        "boxes": detections,
        "risk":  classify_risk(score),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="InfinifyX Inference Server")

yolo_model = None
gnn_model  = None
gnn_device = None


@app.on_event("startup")
async def startup():
    global yolo_model, gnn_model, gnn_device
    for label, path in [("YOLO weight", YOLO_WEIGHT), ("GNN weight", GNN_WEIGHT)]:
        if not path.exists():
            print(f"[server] WARNING: {label} not found at {path}")
    yolo_model             = load_yolo(YOLO_WEIGHT)
    gnn_model, gnn_device  = load_gnn(GNN_WEIGHT)
    print("[server] All models loaded — ready to serve.")


# Serve static files (index.html, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = ws.client
    print(f"[server] Client connected  {client}")
    try:
        while True:
            data  = await ws.receive_bytes()
            frame = bytes_to_frame(data)
            if frame is None:
                continue
            result = infer_frame(yolo_model, gnn_model, gnn_device, frame)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print(f"[server] Client disconnected  {client}")
    except Exception as exc:
        print(f"[server] Error: {exc}")
        try:
            await ws.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=str(SERVER_DIR),
    )
