"""
Data Processing Pipeline
========================
Integrated copy of MLOps/DataProcessing/extract_convert_upload_pipeline.py,
adapted for the Product pipeline directory.

Responsibilities:
  1. Extract a configurable percentage subset from the raw BDD100K dataset.
  2. Convert BDD JSON annotations to YOLO format.
  3. Upload the processed dataset to ClearML as a versioned Dataset artifact.

ClearML project: MLOps_Product_Assisted_Driving
Execution queue: data_engineer
"""

import os
import sys
import shutil
import json
import random
from clearml import PipelineDecorator, Dataset

# ==========================================
# 0. Secure Configuration Loading
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
local_config_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'clearml.conf'))

is_ci = os.environ.get('CI', 'false').lower() == 'true'

if not is_ci:
    if os.path.exists(local_config_path):
        os.environ['CLEARML_CONFIG_FILE'] = local_config_path
        print(f"[*] Loaded secure ClearML configuration from: {local_config_path}")
    else:
        print(f"[!] Warning: Local clearml.conf not found at {local_config_path}.")
else:
    print("[*] Running in CI mode. Using environment variables for ClearML credentials.")


# ==========================================
# 1. Preprocessing Component
# ==========================================
@PipelineDecorator.component(execution_queue="data_engineer")
def data_preprocessing_step(dataset_path: str, subset_percentage: float) -> str:
    import os
    import json
    import random
    import shutil
    import sys

    # Get the output dir from args or default to next to the original dataset if not running from CLI?
    # Actually, we can just hardcode the output dir relative to the CWD for pipeline consistency
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, "data_preprocessing", "datasets", "bdd100k_subset_yolo")
    DEFAULT_PCT = 0.1
    try:
        if subset_percentage is None or subset_percentage == "" or str(subset_percentage).lower() == "none":
            subset_percentage = DEFAULT_PCT
            print(f"[WARN] subset_percentage was None/empty — defaulting to {DEFAULT_PCT}")
        subset_percentage = float(subset_percentage)
        if subset_percentage <= 0:
            print(f"[WARN] subset_percentage <= 0 — defaulting to {DEFAULT_PCT}")
            subset_percentage = DEFAULT_PCT
    except (TypeError, ValueError) as e:
        print(f"[WARN] Failed to parse subset_percentage={subset_percentage!r}: {e} — defaulting to {DEFAULT_PCT}")
        subset_percentage = DEFAULT_PCT

    print(f"Processing dataset from: {dataset_path}")
    print(f"Extracting {subset_percentage}% of the data...")

    output_dir = os.path.join(os.path.dirname(dataset_path), "bdd100k_subset_yolo")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)

    # BDD100K standard class mapping
    class_mapping = {
        "pedestrian":    0,
        "rider":         1,
        "car":           2,
        "truck":         3,
        "bus":           4,
        "train":         5,
        "motorcycle":    6,
        "bicycle":       7,
        "traffic light": 8,
        "traffic sign":  9,
    }

    # Write classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cls_name, _ in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{cls_name}\n")

    # Locate annotation JSON (BDD100K format)
    train_json = os.path.join(dataset_path, "train", "annotations",
                              "bdd100k_labels_images_train.json")
    train_img_dir = os.path.join(dataset_path, "train", "images")

    if not os.path.exists(train_json):
        json_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        if json_files:
            train_json = json_files[0]
            print(f"Using JSON file found at: {train_json}")
        else:
            print("No JSON label file found. Assuming YOLO format or empty dataset.")
            return output_dir

    print(f"Loading annotations from {train_json}...")
    with open(train_json, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Unsupported JSON format. Expected a list of image annotations.")
        return output_dir

    sample_size = max(1, int(len(data) * (subset_percentage / 100.0)))
    sampled_data = random.sample(data, sample_size)
    print(f"Sampled {sample_size} images out of {len(data)}")

    processed_count = 0
    for item in sampled_data:
        img_name = item.get("name")
        labels   = item.get("labels", [])

        img_src_path = None
        for root, _, files in os.walk(dataset_path):
            if img_name in files:
                img_src_path = os.path.join(root, img_name)
                break

        if not img_src_path:
            continue

        shutil.copy2(img_src_path,
                     os.path.join(output_dir, 'images', 'train', img_name))

        img_w, img_h = 1280.0, 720.0  # BDD100K default resolution
        label_file = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(output_dir, 'labels', 'train', label_file), 'w') as f_out:
            for label in labels:
                category = label.get("category")
                if category not in class_mapping:
                    continue
                cls_id = class_mapping[category]
                box2d  = label.get("box2d")
                if not box2d:
                    continue
                x1, y1, x2, y2 = (float(box2d["x1"]), float(box2d["y1"]),
                                   float(box2d["x2"]), float(box2d["y2"]))
                x_c = ((x1 + x2) / 2.0) / img_w
                y_c = ((y1 + y2) / 2.0) / img_h
                w   = (x2 - x1) / img_w
                h   = (y2 - y1) / img_h
                x_c, y_c, w, h = (max(0, min(1, v)) for v in (x_c, y_c, w, h))
                f_out.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        processed_count += 1

    import yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_data = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": 10,
        "names": ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
    }
    with open(yaml_path, 'w') as f_yaml:
        yaml.safe_dump(yaml_data, f_yaml, sort_keys=False)

    print(f"Processed and converted {processed_count} images to YOLO format.")
    return output_dir


# ==========================================
# 2. Upload Component
# ==========================================
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def data_uploading_step(processed_dataset_path: str, dataset_name: str) -> str:
    from clearml import Dataset
    import os

    dataset_project = "MLOps_Product_Assisted_Driving"

    print(f"Creating ClearML Dataset: {dataset_name} in project {dataset_project}")
    dataset = Dataset.create(
        dataset_project=dataset_project,
        dataset_name=dataset_name,
    )

    print(f"Adding files from {processed_dataset_path}")
    dataset.add_files(path=processed_dataset_path)

    print("Uploading to ClearML server...")
    dataset.upload()
    dataset.finalize()

    print(f"Upload complete! Dataset ID: {dataset.id}")
    return dataset.id


# ==========================================
# 3. Pipeline Stitching
# ==========================================
@PipelineDecorator.pipeline(
    name='Data_Processing_Pipeline',
    project='MLOps_Product_Assisted_Driving',
    version='1.0',
    add_pipeline_tags=False,
    default_queue="data_engineer",
)
def data_processing_pipeline(
    dataset_path: str,
    subset_percentage: float,
    output_dataset_name: str,
    skip_upload: bool = True,
):
    """
    Full data processing pipeline:
      Step 1 — data_preprocessing_step:
          Extracts a subset from BDD100K and converts JSON → YOLO format.
      Step 2 — data_uploading_step:
          Uploads the processed dataset to ClearML as a versioned artifact.
    """
    processed_path = data_preprocessing_step(
        dataset_path=dataset_path,
        subset_percentage=subset_percentage,
    )
    if skip_upload:
        print("[skip_upload] Skipping dataset upload (skip_upload=True)")
        return processed_path
    dataset_id = data_uploading_step(
        processed_dataset_path=processed_path,
        dataset_name=output_dataset_name,
    )
    return dataset_id


# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == '__main__':
    from clearml import Task as _Task

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data_preprocessing/datasets/bdd100k")
    parser.add_argument("--subset_percentage", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="data_preprocessing/datasets/bdd100k_subset_yolo")
    args = parser.parse_args()

    # Always use the CLI args
    ds_path = args.dataset_path
    percentage = args.subset_percentage
    ds_name = f"BDD100k_{percentage}percent_YOLO"

    # If the user provides a relative path, resolve it relative to the project root
    if not os.path.isabs(ds_path):
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        ds_path = os.path.join(project_root, ds_path)

    print(f"[*] Dataset path  : {ds_path}")
    print(f"[*] Output dir    : {args.output_dir}")
    print(f"[*] Subset        : {percentage}%")
    print("[*] Running pipeline locally.")
    
    PipelineDecorator.run_locally()
    data_processing_pipeline(
        dataset_path=ds_path,
        subset_percentage=percentage,
        output_dataset_name=ds_name,
        skip_upload=True,
    )
