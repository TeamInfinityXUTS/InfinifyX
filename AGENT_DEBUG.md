# AGENT_DEBUG.md

## ClearML Security Configuration Update

**Date**: 2026-05-19
**Context**: Replaced hardcoded ClearML credentials in the codebase with a centralized, secure configuration file (`clearml.conf`).

### Actions Taken:

1. **Created Root Configuration**:
   - Ensured a centralized `clearml.conf` exists at the root of the project (`d:\UTS\2026Autumn\42174 Artificial Intelligence Studio\Infinity\InfinifyX\clearml.conf`) holding `web_server`, `api_server`, `files_server`, `access_key`, and `secret_key`.

2. **Refactored `MLOps` Pipelines**:
   - Modified `MLOps/DataProcessing/upload_pipeline.py` and `MLOps/DataProcessing/data_augument_pipeline.py`.
   - Updated the scripts to dynamically load the centralized `clearml.conf` by setting the `CLEARML_CONFIG_FILE` environment variable via `os.environ`.
   - Fixed path resolutions to accurately point to the project root using `os.path.abspath(os.path.join(current_dir, '..', '..', 'clearml.conf'))`.

3. **Refactored `data_preprocessing` Notebooks**:
   - Automatically updated `upload_data_clearml.ipynb`, `remote_data_preprocessing-Copy1.ipynb`, `upload_augument_dataset.ipynb`, and `data_preprocessing.ipynb`.
   - Removed explicit Jupyter magic commands containing credentials (`%env CLEARML_API_ACCESS_KEY=...`).
   - Injected a secure cell that sets up `os.environ['CLEARML_CONFIG_FILE']` using relative pathing (`../clearml.conf`).
   - Ran `jupyter nbconvert --clear-output` to strip all executed cell outputs from the notebooks to eliminate residue of the hardcoded secret keys.

4. **Security Verification**:
   - Conducted a directory-wide regex search for `CLEARML_API_ACCESS_KEY` to verify zero exposure within the `MLOps` and `data_preprocessing` branches.

### Current Status:
- All data preprocessing notebooks and MLOps scripts are synchronized and exclusively rely on the root `clearml.conf` for connecting to the ClearML platform securely.
- Environment variables are set safely without leaving a permanent trace in the version-controlled codebase.

---

## Data CI Pipeline Implementation

**Date**: 2026-05-19
**Context**: Created a Continuous Integration (CI) pipeline for data preprocessing and ingestion to automate dataset subset extraction, format validation/conversion, and ClearML publishing.

### Actions Taken:

1. **Created New Pipeline Script**:
   - Path: `MLOps\DataProcessing\extract_convert_upload_pipeline.py`
   - Designed to act as the primary Data CI entry point for the `MLOps_Level2` project.

2. **Implemented ClearML `@PipelineDecorator` Architecture**:
   - Built a modular pipeline splitting the workflow into discrete, independently tracked components.
   - **`data_preprocessing_step` (Component 1)**: 
     - Extracts a dynamic percentage (default 1%) of a large dataset (e.g., BDD100K).
     - Automatically parses raw JSON annotations (BDD100k format), normalizes bounding box coordinates, maps classes to standard IDs, and generates YOLO format `.txt` label files.
   - **`data_uploading_step` (Component 2)**: 
     - Takes the standardized YOLO dataset directory.
     - Registers a new `clearml.Dataset` artifact and uploads it to the ClearML server.
   - **`extract_and_upload_pipeline` (Pipeline Stitching)**: 
     - Links the two components.
     - Hardcoded to run tasks on the `data_engineer` execution queue, automatically assigning them to the active local worker.

3. **Secure Integration**:
   - Integrated the dynamic `clearml.conf` secure loading logic at the head of the pipeline to ensure automated runs authenticate safely.

---

## Feature Engineering Pipeline Implementation

**Date**: 2026-05-20
**Context**: Created a standalone ClearML feature engineering pipeline that converts raw image data into a pre-built graph dataset (`graph_cache.pt`) for downstream GNN training, using the custom-trained YOLOv8 model as the feature backbone.

### Actions Taken:

1. **Created New Pipeline Script**:
   - Path: `MLOps/FeatureEngineering/feature_engineering_pipeline.py`
   - Registered under ClearML project `MLOps_Level2` with pipeline name `Feature_Engineering_Pipeline`.
   - Both steps dispatch to the `data_engineer` execution queue.
   - Pipeline controller runs locally via `PipelineDecorator.run_locally()` (avoids requiring a `services` queue).

2. **Implemented ClearML `@PipelineDecorator` Architecture**:
   - **`graph_build_step` (Component 1, `cache=True`)**:
     - Inputs: `yolo_weight_path` (`models/yolo/Yolov8_best.pt`), `yolo_cache_path` (`models/yolo/yolo_cache_train.pt`), `data_dir` (image folder).
     - Injects custom attention modules `CBAM` and `SE` into `ultralytics.nn.tasks` before loading the model — required because `Yolov8_best.pt` was trained with these custom blocks.
     - Runs YOLO inference on every image; extracts 5-dimensional node features `[cx, cy, conf, dist, cls]` per detected object.
     - Constructs a spatial proximity graph per frame (`torch_geometric.data.Data`): nodes = detections, edges = pairs with normalized distance < 0.3.
     - Computes a per-frame risk score weighted by class risk, confidence, spatial centrality, and object size.
     - Saves the resulting `list[(Data, risk_score)]` as `models/gnn/graph_cache.pt`.
   - **`upload_graph_cache_step` (Component 2, `cache=False`)**:
     - Uploads `graph_cache.pt` as a versioned ClearML artifact attached to the pipeline task under `MLOps_Level2`.

3. **Model Weight Organisation**:
   - Relocated `GNN_best.pt` from repo root → `models/gnn/GNN_best.pt` via `git mv`.
   - `models/gnn/` now consolidates all GNN-related artifacts: `GNN_best.pt` and `graph_cache.pt`.

4. **Bug Fixes Applied**:
   - Fixed corrupted first line (`yolo_cache_trainimport os`) and wrong `CACHE_PATH = ".pt"` in `MLOps/Model_Training_Evaluation_and_Deployment_Pipeline/YOLOFeatureExtractor.py`.

5. **Data Directory Configuration**:
   - Default `data_dir` set to `data_preprocessing/datasets/bdd100k_subset_yolo/images/train/` (698 images) for fast initial test runs.
   - Full run option: `data_preprocessing/datasets/bdd100k/train/images/` (70,000 images).

### Running the Pipeline:
```bash
# From the project root with the venv activated:
python MLOps\FeatureEngineering\feature_engineering_pipeline.py
```

### Current Status:
- Pipeline runs successfully end-to-end on the local `data_engineer` agent.
- Output: `models/gnn/graph_cache.pt` — a list of `(torch_geometric.data.Data, torch.Tensor)` tuples ready for GNN training.
- Artifact is versioned and accessible via ClearML UI under `MLOps_Level2 / Feature_Engineering_Pipeline`.

---

## Real-Time Inference Server

**Date**: 2026-05-21
**Context**: Created a FastAPI-based inference server that exposes a WebSocket endpoint for real-time, frame-by-frame risk detection using `Yolov8_best.pt` and `risk_model.pt` (GNN). A browser client captures the local camera and overlays bounding boxes and risk level on a live video feed.

### Architecture:
```
Browser Client (server/static/index.html)
  getUserMedia() → JPEG frame → WebSocket send
  ← JSON {boxes, risk} → Canvas overlay + risk banner

FastAPI Server (server/inference_server.py)
  GET /          → serves index.html
  WebSocket /ws  → receives JPEG bytes → YOLO → GNN → JSON response
```

### Actions Taken:

1. **Created `server/inference_server.py`**:
   - Loads `models/yolo/Yolov8_best.pt` at startup with `CBAM` and `SE` injected into `ultralytics.nn.tasks.__dict__` before calling `YOLO()` — identical pattern to training pipeline.
   - Loads `models/gnn/risk_model.pt` into `SpatioTemporalModel` (2-layer GCN → global mean pool → MLP → scalar). Checkpoint format `{"model_state_dict": ...}` and raw state dict both supported.
   - WebSocket handler at `/ws`: receives raw JPEG bytes → `cv2.imdecode` → `run_yolo()` → `build_graph()` → `gnn(graph)` → `classify_risk()` → sends JSON.
   - `build_graph()` uses the same 5-dim node features `[cx, cy, conf, dist, cls]` and proximity threshold `< 0.3` as the feature engineering pipeline, ensuring consistency.
   - Risk thresholds: score `< 0.33` → Low, `< 0.66` → Medium, `≥ 0.66` → High.
   - Static files served from `server/static/` at `GET /`.

2. **Created `server/static/index.html`** (single-file browser client):
   - Requests camera via `getUserMedia({ video: {width:1280, height:720} })`.
   - Captures frames to a hidden `<canvas>`, exports as JPEG blob, sends binary over WebSocket at 10 FPS.
   - On message received: parses JSON, draws bounding boxes with class label + confidence on an overlay `<canvas>`, updates the risk banner colour (green/yellow/red).
   - Displays live FPS, detection count, and raw risk score in a stats bar.

3. **Model Weight Organisation**:
   - `risk_model.pt` was already relocated to `models/gnn/risk_model.pt` (confirmed via `git ls-files`).
   - Final `models/` layout:
     ```
     models/
     ├── yolo/
     │   ├── Yolov8_best.pt
     │   └── yolo_cache_train.pt
     └── gnn/
         ├── GNN_best.pt
         ├── risk_model.pt
         └── graph_cache.pt
     ```

### Running the Server:
```powershell
# Install dependencies (once)
pip install fastapi "uvicorn[standard]" opencv-python torch-geometric

# Start server from project root
.venv\Scripts\Activate.ps1
python server\inference_server.py

# Open in browser
Start-Process "http://localhost:8000"
```

### Current Status:
- Server loads both models at startup and serves the browser client at `http://localhost:8000`.
- WebSocket inference pipeline: JPEG frame → YOLO detections → spatial graph → GNN risk score → JSON response.
- Browser client overlays bounding boxes and a colour-coded risk banner (Low/Medium/High) in real time.
