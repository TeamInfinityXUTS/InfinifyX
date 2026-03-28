from ultralytics import YOLO
from Model_featureExtraction import FeatureExtractor
from Risk_prediction import RiskPredictionModel

import cv2
import numpy as np


class Warning_System:
    def __init__(self, yolo_model, risk_model):
        self.detector = YOLO(yolo_model)
        self.extractor = FeatureExtractor()
        self.risk_model = RiskPredictionModel()
        self.risk_model.load(risk_model)

        self.labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        self.colors = {
            # green
            0: (0, 255, 0),
            # yellow
            1: (0, 255, 255),
            # red
            2: (0, 0, 255)
        }

    def run(self, image_path, save_path = "output.jpg"):
        img = cv2.imread(image_path)
        results = self.detector(image_path)
        detections = []

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = self.detector.names[cls_id]

                detections.append({
                    "box": xyxy,
                    "class": class_name,
                    "confidence": conf
                })

                x1, y1, x2, y2, = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{class_name}{conf: .2f}",
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 0, 0), 2)
        features = self.extractor.extract(detections)
        risk_vector = self.extractor.build_risk_vector(features)
        risk_vector = risk_vector.reshape(1, -1)

        pred = self.risk_model.predict(risk_vector)[0]
        prob = self.risk_model.predict_proba(risk_vector)[0]

        risk_label = self.labels[pred]
        color = self.colors[pred]

        text = f"Risk: {risk_label} ({prob[pred]: .2f})"
        cv2.putText(img, text,
                   (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, color, 3)

        cv2.imwrite(save_path, img)

        print("==============================")
        print("Risk level:", risk_label)
        print("Probability:", prob)
        print("Saved:", save_path)
        print("==============================")

        return risk_label, prob
