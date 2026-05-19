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
