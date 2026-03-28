import os
import numpy as np
from ultralytics import YOLO
from Model_featureExtraction import FeatureExtractor
from Risk_prediction import RiskPredictionModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from clearml import Task
import matplotlib.pyplot as plt


task = Task.init(project_name="Risk_Prediction_System", task_name="YOLO_Risk_Model")

MODEL_PATH = "runs/detect/runs/YOLOv8_BDD100K/experiment_20260323_050256/weights/best.pt"
TEST_DIR = "bdd100k_full_data/test"
SAVE_DIR = "risk_dataset7"
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

X_np = np.array(all_risk_vectors)
num_objects_75 = np.percentile(X_np[:,1], 75)
weighted_risk_75 = np.percentile(X_np[:,3], 75)
avg_distance_25 = np.percentile(X_np[:,0], 25)

def generate_risk_label(risk_vector):
    avg_distance, num_objects, avg_conf, weighted_risk = risk_vector

    if num_objects >= num_objects_75 or weighted_risk >= weighted_risk_75 or avg_distance <= avg_distance_25:
        return 2
    elif num_objects >= 2 or weighted_risk >= 0.5:
        return 1
    else:
        return 0

y = np.array([generate_risk_label(v) for v in X])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Labels distribution (train): {np.bincount(y_train)}, (test): {np.bincount(y_test)}")


plt.hist(y_train, bins=3, alpha=0.5, label="Train")
plt.hist(y_test, bins=3, alpha=0.5, label="Test")
plt.title("Risk Label Distribution")
plt.legend()
plt.show()


risk_model = RiskPredictionModel()
risk_model.train(X_train, y_train)
print("Risk Model trained!")

y_pred = risk_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nEvaluation Results")
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

model_path = os.path.join(SAVE_DIR, "risk_model.pkl")
risk_model.save(model_path)
task.upload_artifact("risk_model", model_path)

np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
print("\nModel and datasets saved!")
