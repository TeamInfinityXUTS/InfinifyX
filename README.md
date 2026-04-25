# InfinifyX: Urban Driving Assistance System

**InfinifyX** is a camera-based assistance system designed to enhance urban road safety. By leveraging computer vision and real-time inference, it provides low-cost, accessible safety warnings for everyday drivers, detecting potential hazards and environmental conditions in real-time.

## 🚀 Project Pillars
* **Real-Time Detection:** Processing live camera feeds to identify key road objects (vehicles, pedestrians, lanes).
* **Risk Inference:** Applying logic to evaluate immediate environmental hazards.
* **Warning Support:** Delivering intuitive and timely feedback to the driver for safer decision-making.


---

## 🛠 MLOps & Pipelines
To ensure model robustness and reproducible data engineering, this project utilizes **ClearML** for pipeline orchestration:

* **Dataset Upload Pipeline:** Automates the ingestion of raw urban driving data into centralized versioned storage.
* **Data Augmentation Pipeline:** A self-contained workflow that applies geometric and pixel-level transformations (Albumentations) to YOLO-format datasets, automatically tracking lineage and generating augmented versions for model training.

---

## 📋 Key Assumptions
* **Accessibility:** Optimized for standard camera hardware and consumer-grade mobile devices/computers.
* **Environment:** Specifically tuned for standard urban driving scenarios (city roads, intersections).
* **Robustness:** System performance is designed to account for variable lighting, weather conditions, and hardware quality.

---

### 💡 Recent Updates
* Integrated **ClearML Pipeline Controller** for automated data workflows.
* Implemented automated **YOLO-format augmentation** with safe coordinate clipping.
* Standardized environment configuration using local `.conf` file management for enhanced security.