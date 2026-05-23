# Project Phase Summary: MLOps Pipeline Integration, Real-Time Inference, and CI/CD Automation

This document outlines the major tasks, implementations, and system integrations completed during this development phase for the **InfinifyX** project.

---

## 1. CI/CD Workflows (`.github/workflows/`)
To automate validation, training, and deployment, two main workflows were established:

*   **Data CI Pipeline (`data_ci_pipeline.yml`)**:
    *   **Trigger**: Triggers automatically on pushes to the `main` branch when changes are detected in critical directories: `Product/product_piepline/**`, `MLOps/**`, or the workflow file itself.
    *   **Functionality**: Automatically initiates an end-to-end ClearML pipeline (`Product/product_piepline/end_to_end_pipeline.py`) running on the local execution queue (`data_engineer`). It is optimized for CI with a lightweight subset (0.1% of data) and a single epoch to ensure fast verification.
*   **CD Deployment Pipeline (`cd_deploy.yml`)**:
    *   **Trigger**: Manual trigger (`workflow_dispatch`) with a safety gate requiring the user to type `"deploy"` to proceed.
    *   **Functionality**: Performs syntax validation (`py_compile`) on the backend server (`server/inference_server.py`) and verifies the existence of deployment scripts. Once validation passes, it outputs detailed instructions for manual server restarts on SageMaker.

---

## 2. MLOps Pipeline Enhancements (`MLOps/`)
The data-engineering and training pipelines were modularized and connected to ClearML tracking:

*   **Data Ingestion & Processing Pipeline (`MLOps/DataProcessing/extract_convert_upload_pipeline.py`)**:
    *   Implements the `@PipelineDecorator` structure to process the BDD100K raw dataset.
    *   **Data Extraction & Conversion**: Dynamically extracts a configurable subset (e.g., 1%), parses raw JSON labels, normalizes bounding boxes, and exports standard YOLO-format annotations.
    *   **Data Ingestion**: Standardizes directories, packages them, and registers/uploads them as a versioned `clearml.Dataset`.
*   **Feature Engineering Pipeline (`MLOps/FeatureEngineering/feature_engineering_pipeline.py`)**:
    *   Constructs a spatial-proximity graph (`graph_cache.pt`) from raw images using the custom-trained YOLOv8 model as a feature extractor.
    *   Dynamically injects attention modules (`CBAM` and `SE` blocks) into the YOLO load sequence to match the custom-trained model architecture.
    *   Extracts 5D object node features `[cx, cy, confidence, distance, class_id]` and builds spatial edges for objects within normalized proximity (`distance < 0.3`).
    *   Calculates relative danger indices per frame and uploads the serialized graph cache as a versioned pipeline artifact.

---

## 3. Product Pipeline Refactoring (`Product/product_piepline/`)
Code components inside `Product/product_piepline/` were refactored, extended, and integrated to coordinate with MLOps pipelines:
*   Refactored core files: `data_processing_pipeline.py`, `feature_engineering_pipeline.py`, `gnn_training_evaluation_pipeline.py`, `model_hyper_parameter_tuning.py`, `multi_model_training_and_model_selection.py`, `yolo_training_evaluation_pipeline.py`, and the main coordinator `end_to_end_pipeline.py`.
*   Decoupled hardcoded paths and ensured all configuration paths default to standard workspace boundaries, enabling reproducible runs on different worker agents.

---

## 4. Real-Time Inference Server & Deployment Helper (`server/` & `scripts/`)
We bridged the gap between model training and real-world deployment by implementing a real-time web client and hosting backend:

*   **FastAPI Backend (`server/inference_server.py`)**:
    *   Initializes the system by loading `models/yolo/Yolov8_best.pt` (injecting CBAM/SE attention blocks at runtime) and `models/gnn/risk_model.pt` (SpatioTemporal GNN).
    *   Exposes a low-latency WebSocket endpoint (`/ws`) that receives JPEG byte frames, runs object detection, converts detections into proximity graphs, performs GNN risk inference, and returns frame metadata.
*   **Browser-Based Client (`server/static/index.html`)**:
    *   Accesses the user's camera feed at 1280x720 via `getUserMedia` and streams video frames over WebSocket at 10 FPS.
    *   Renders real-time bounding boxes, confidence ratings, and color-coded risk levels (Green for Low, Yellow for Medium, Red for High) directly onto an overlay canvas.
*   **Deployment Helper (`scripts/deploy.sh`)**:
    *   Simplifies deployments on SageMaker down to a single execution:
        ```bash
        cd /home/sagemaker-user/InfinifyX && bash scripts/deploy.sh
        ```
    *   Automates pulling latest main-branch code, killing any active `inference_server` processes, starting the FastAPI app in the background (`nohup`), and verifying server responsiveness on port 8000.

---

## 5. Agent Governance and Logging (`AGENT.md` & `AGENT_DEBUG.md`)
To maintain consistency and security during development, the agent workflow was standardized:

*   **Agent Guidelines (`AGENT.md`)**:
    *   Specifies boundaries to prevent agents from modifying template assets in `AI-Studio-ClearML/`.
    *   Details environment security protocols, particularly centralized config management via `clearml.conf` (preventing hardcoded API keys in commits/notebooks).
*   **Development Changelog (`AGENT_DEBUG.md`)**:
    *   Tracks detailed debugging steps, credential migrations, GNN model state dict loaders fixes, and code modifications over time to maintain traceability.

---

## 6. AWS SageMaker & ClearML Integration Status
*   **ClearML**: All pipelines are fully integrated and trace-logged on ClearML. Data ingestion, feature extraction, and model training execution metrics are safely recorded.
*   **Semi-Automated CD Constraints**:
    *   Due to the restricted nature of the school-issued AWS SageMaker account, remote invocation via AWS Systems Manager (SSM) was blocked by the following permissions constraint:
        ```
        aws: [ERROR]: An error occurred (AccessDeniedException) when calling the DescribeInstanceInformation operation: User: arn:aws:sts::443142193439:assumed-role/42174-AIstudio-Aut26-sagemaker-execution-role/SageMaker is not authorized to perform: ssm:DescribeInstanceInformation on resource: arn:aws:ssm:ap-southeast-2:443142193439:* because no identity-based policy allows the ssm:DescribeInstanceInformation action
        ```
    *   **Resolution**: Implemented a robust semi-automated pipeline. GitHub Actions manages linting and compile validations, while `scripts/deploy.sh` serves as an easy, one-line manual execution utility inside the SageMaker terminal.
*   **Access & Verification**:
    *   The deployment script has been executed successfully on SageMaker.
    *   The application is fully operational and accessible via the SageMaker JupyterLab proxy URL:
        ```
        https://tzapmkixqvdpkjz.studio.sagemaker.ap-southeast-2.app.aws/jupyterlab/default/proxy/8000/
        ```
