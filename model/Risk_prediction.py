import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class RiskPredictionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=200, random_state=42)
        self.is_trained = False

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("Risk model trained")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained or loaded!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path="risk_model.pkl"):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler
        }, path)

    def load(self, path="risk_model.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = True
