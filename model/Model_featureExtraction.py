import numpy as np
import cv2


class FeatureExtractor:
    def __init__(self, img_width=1280, img_height=720):
        self.img_width = img_width
        self.img_height = img_height

        self.class_weights = {
            'person': 1.0,
            'car': 0.8,
            'bus': 0.9,
            'truck': 1.0,
            'bike': 0.85,
            'motor': 0.9,
            'traffic light': 0.3,
            'traffic sign': 0.2
        }

    def get_center(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def get_area(self, box):
        x1, y1, x2, y2 = box
        return max(x2 - x1, 0) * max(y2 - y1, 0)

    def estimate_distance(self, box):
        area = self.get_area(box)
        if area == 0:
            return float("inf")
        distance = np.sqrt(1 / (area / (self.img_width * self.img_height) + 1e-6))
        return distance

    def in_danger_zone(self, cx, cy):
        center_x_min = self.img_width * 0.3
        center_x_max = self.img_width * 0.7
        center_y_min = self.img_height * 0.4
        center_y_max = self.img_height * 0.9
        return (center_x_min < cx < center_x_max) and (center_y_min < cy < center_y_max)

    def extract(self, detections):
        features = []
        for det in detections:
            box = det["box"]
            cls = det["class"]
            conf = det["confidence"]

            cx, cy = self.get_center(box)
            area = self.get_area(box)
            distance = self.estimate_distance(box)
            danger_zone = self.in_danger_zone(cx, cy)
            class_weight = self.class_weights.get(cls, 0.5)

            feature = {
                "class": cls,
                "confidence": conf,
                "center_x": cx / self.img_width,
                "center_y": cy / self.img_height,
                "area": area / (self.img_width * self.img_height),
                "distance": distance,
                "in_danger_zone": int(danger_zone),
                "class_weight": class_weight
            }
            features.append(feature)
        return features

    def build_risk_vector(self, features):
        if len(features) == 0:
            return np.zeros(4)

        avg_distance = np.mean([f["distance"] for f in features])
        num_objects = len(features)
        avg_conf = np.mean([f["confidence"] for f in features])
        weighted_risk = np.mean([f["class_weight"] / (f["distance"] + 1e-6) for f in features])

        return np.array([
            avg_distance,
            num_objects,
            avg_conf,
            weighted_risk
        ])
