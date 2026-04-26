import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from ultralytics import YOLO
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from clearml import Task

DATA_CACHE_PATH = "graph_cache_train.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_graph(node_features):
    x = torch.tensor(node_features, dtype=torch.float, device=device)

    edges = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                dist = torch.dist(x[i, :2], x[j, :2])
                if dist < 0.3:
                    edges.append([i, j])

    if len(edges) == 0:
        edges = [[0, 0]]

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def compute_risk_score(dets):
    CLASS_RISK = {
        0: 0.9, 1: 0.6, 2: 0.8, 3: 0.85,
        4: 0.7, 5: 0.75, 6: 0.95,
        7: 0.4, 8: 0.3
    }

    score = 0.0

    for d in dets:
        cls = int(d["cls"])
        conf = float(d["confidence"])

        x1, y1, x2, y2 = d["box"]

        cx = (x1 + x2) / 2 / 1280
        cy = (y1 + y2) / 2 / 720

        spatial = 1 - ((cx - 0.5)**2 + (cy - 0.5)**2) ** 0.5

        area = (x2 - x1) * (y2 - y1)
        size = min(area / (1280 * 720), 1.0)

        score += CLASS_RISK.get(cls, 0.5) * conf * (0.7 * spatial + 0.3 * size)

    return score / max(len(dets), 1)

class CachedRiskDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, extractor, seq_len=3):

        self.seq_len = seq_len

        print("Building graph dataset...")

        images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.data = []

        for i in tqdm(range(len(images))):
            path = os.path.join(image_dir, images[i])

            det = extractor.batch_infer([path])[0]
            feat = extractor.extract(det)

            graph = build_graph(feat)
            label = compute_risk_score(det)

            self.data.append((graph, torch.tensor(label, dtype=torch.float32)))

        print(f"Dataset built: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
