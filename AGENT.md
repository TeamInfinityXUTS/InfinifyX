# InfinifyX - Agent Guidelines

This document serves as a guide for AI agents and developers working on the **InfinifyX** project. 

## 1. Project Overview
InfinifyX is a camera-based assistance system designed to enhance urban road safety. It leverages computer vision and real-time inference to provide safety warnings for everyday drivers by detecting potential hazards and environmental conditions in real-time.

**Note on Templates:** The `AI-Studio-ClearML/` directory is a cloned template provided for reference purposes. Agents **should NOT modify** files within this template directory. All actual project development, modifications, and bug fixes must be performed within the `data_preprocessing/`, `MLOps/`, and `model/` directories.

## 2. Core Directories to Modify

### `data_preprocessing/`
This directory contains the initial data ingestion, exploratory analysis, and local processing scripts.
- **Purpose**: Managing raw datasets (e.g., BDD100K), running data augmentation experiments, and uploading data to ClearML.
- **Key Files**: 
  - Jupyter Notebooks (`data_preprocessing.ipynb`, `data_augumentation.ipynb`, `upload_data_clearml.ipynb`) for interactive data manipulation.
  - Python scripts (`check_data.py`) and configurations (`bdd100k_tiny.yaml`).

### `MLOps/`
This directory is responsible for the automated pipelines orchestrated via ClearML. It ensures model robustness and reproducible data engineering.
- **`DataProcessing/`**: Contains self-contained scripts to orchestrate data workflows.
  - `upload_pipeline.py`: Automates the ingestion of raw urban driving data into centralized versioned storage via ClearML.
  - `data_augument_pipeline.py`: Applies geometric and pixel-level transformations to YOLO-format datasets.
- **`Model_Training_Evaluation_and_Deployment_Pipeline/`**: Contains the end-to-end model training pipelines.
  - `model_piepline.py`: The main ClearML pipeline decorator script that connects dataset ingestion, YOLO training, Graph construction (`graph_build_step`), Graph Neural Network (GNN) training, evaluation, and model registration.
  - Sub-modules: `yolo_trainer.py`, `gnn_trainer.py`, `YOLOFeatureExtractor.py`, and attention mechanisms (`CBAM.py`, `SE.py`).

### `model/`
This directory holds the core inference logic, standalone training scripts, and the real-time warning system utilized by the application.
- **Purpose**: Defines how the model makes predictions and handles the end-user inference.
- **Key Files**:
  - `warning_system.py`: The main application class (`Warning_System`) that integrates the YOLO detector and a `RiskPredictionModel`. It outputs risk levels (Low, Medium, High) with bounding boxes on image frames.
  - `Model_featureExtraction.py` & `Risk_prediction.py`: Used to extract object features and infer environmental risk.
  - `train_yolo.py`, `Risk_pipeline.py`, `resume_train.py`: Utilities for training models directly outside the ClearML pipeline if needed.

## 3. General Directives for Agents
1. **Focus Area**: Always restrict changes to `data_preprocessing`, `MLOps`, and `model` directories. Use `AI-Studio-ClearML` only as a reference for ClearML API usage or structural patterns.
2. **ClearML Integration**: When updating pipeline components in `MLOps/`, ensure that ClearML `PipelineDecorator` or `PipelineController` logic remains intact.
3. **Environment Configuration**: Always use local `.conf` file management for ClearML authentication. Avoid hardcoding credentials in the scripts.
4. **Documentation**: Maintain descriptive docstrings and comments when updating logic for data pipelines or model architectures.
x
