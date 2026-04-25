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

CACHE_PATH = "yolo_cache_train.pt"

def build_yolo_cache(src_path, cache_path):
    if not os.path.exists(cache_path):
        print("Creating YOLO cache...")
        shutil.copyfile(src_path, cache_path)
    return cache_path



class YOLOFeatureExtractor:
    def __init__(self, model_path):
        model_path = build_yolo_cache(model_path, CACHE_PATH)
        self.model = YOLO(model_path)

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
        nodes = []

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cx = (x1 + x2) / 2 / 1280
            cy = (y1 + y2) / 2 / 720
            conf = d["confidence"]

            area = (x2 - x1) * (y2 - y1)
            dist = 1.0 / (area + 1e-6)

            cls = d["cls"]
            nodes.append([cx, cy, conf, dist, cls])

        if len(nodes) == 0:
            nodes = [[0.5, 0.5, 0.0, 1.0, 0]]

        return np.array(nodes, dtype=np.float32)
