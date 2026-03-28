import os
import numpy as np
from ultralytics import YOLO
from Model_featureExtraction import FeatureExtractor
from Risk_prediction import RiskPredictionModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from clearml import Task


task = Task.init(project_name="Risk_Prediction_System", task_name="YOLO_Risk_Model")


MODEL_PATH = "runs/detect/runs/YOLOv8_BDD100K/experiment_20260323_050256/weights/best.pt"
TEST_DIR = "bdd100k_full_data/test"
SAVE_DIR = "risk_dataset5"
os.makedirs(SAVE_DIR, exist_ok=True)


model = YOLO(MODEL_PATH)
extractor = FeatureExtractor()
all_risk_vectors = []

print("Running inference on test dataset...\n")


for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(TEST_DIR, img_name)
    results = model(img_path)

    detections = []
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "box": xyxy,
                "class": model.names[cls_id],
                "confidence": conf
            })

    features = extractor.extract(detections)
    risk_vector = extractor.build_risk_vector(features)
    all_risk_vectors.append(risk_vector)

    print(f"{img_name} → Risk Vector: {risk_vector}")


X = np.array(all_risk_vectors)
print("\n Dataset created! Shape:", X.shape)

from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)

np.random.seed(42)
noise_train = np.random.normal(0, 0.02, size=X_train_norm.shape[0])
risk_scores_train = (
    0.4*X_train_norm[:,0] + 0.3*X_train_norm[:,1] + 
    0.2*X_train_norm[:,2] + 0.1*X_train_norm[:,3]**2
) + noise_train

q1 = np.quantile(risk_scores_train, 0.33)
q2 = np.quantile(risk_scores_train, 0.66)

y_train = np.array([
    0 if r <= q1 else 1 if r <= q2 else 2
    for r in risk_scores_train
])

noise_test = np.random.normal(0, 0.02, size=X_test_norm.shape[0])
risk_scores_test = (
    0.4*X_test_norm[:,0] + 0.3*X_test_norm[:,1] + 
    0.2*X_test_norm[:,2] + 0.1*X_test_norm[:,3]**2
) + noise_test

y_test = np.array([
    0 if r <= q1 else 1 if r <= q2 else 2
    for r in risk_scores_test
])


print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Labels distribution (train): {np.bincount(y_train)}, (test): {np.bincount(y_test)}")

import matplotlib.pyplot as plt
plt.hist(risk_scores_train, bins=50, alpha=0.5, label="Train")
plt.hist(risk_scores_test, bins=50, alpha=0.5, label="Test")
plt.axvline(q1, color='r', linestyle='--', label='Q1')
plt.axvline(q2, color='g', linestyle='--', label='Q2')
plt.title("Risk Scores Distribution")
plt.legend()
plt.show()


risk_model = RiskPredictionModel()
risk_model.train(X_train, y_train)
print(" Risk Model trained!")


y_pred = risk_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n Evaluation Results")
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


model_path = os.path.join(SAVE_DIR, "risk_model.pkl")
risk_model.save(model_path)
task.upload_artifact("risk_model", model_path)

logger = task.get_logger()
logger.report_scalar("Accuracy", "test", acc, iteration=0)


np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
print("\n Model and datasets saved!")
