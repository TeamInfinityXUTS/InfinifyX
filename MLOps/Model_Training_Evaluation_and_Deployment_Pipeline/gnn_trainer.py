import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from YOLOFeatureExtractor import YOLOFeatureExtractor
from CachedRiskDataset import build_graph, compute_risk_score
from clearml import Task
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpatioTemporalModel(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()

        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, hidden)

        self.risk_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, g):

        x = F.relu(self.gcn1(g.x, g.edge_index))
        x = F.relu(self.gcn2(x, g.edge_index))
        x = global_mean_pool(x, g.batch)

        return self.risk_head(x)
    


class Trainer:
    def __init__(self, yolo_path):

        self.extractor = YOLOFeatureExtractor(yolo_path)
        self.model = SpatioTemporalModel().to(device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def train(self, train_set, val_set, epochs=1):

        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

        os.makedirs("GNN_checkpoints", exist_ok=True)
        best_loss = float("inf")

        for epoch in range(epochs):

            total_loss = 0.0

            self.model.train()

            for batch, labels in tqdm(train_loader):

                batch = batch.to(device)

                labels = labels.to(device).float().view(-1)

                # forward
                preds = self.model(batch).view(-1)

                loss = self.loss_fn(preds, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()

            if len(train_loader) == 0:
                print("Empty train loader")
                return

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1} | Loss {avg_loss:.6f}")

            # =========================
            # save last
            # =========================
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "loss": avg_loss
            }, "GNN_checkpoints/last.pt")

            # =========================
            # save best
            # =========================
            if avg_loss < best_loss:
                best_loss = avg_loss

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    "loss": avg_loss
                }, "GNN_checkpoints/best.pt")

                print(f"Best model saved (loss={best_loss:.6f})")
