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

# In CI environments (e.g., GitHub Actions), credentials come from environment
# variables (Secrets), so clearml.conf is not needed.
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
@PipelineDecorator.component(cache=True, execution_queue="data_engineer")
def data_preprocessing_step(dataset_path: str, subset_percentage: int) -> str:
    import os
    import sys
    import shutil
    import json
    import random
    
    print(f"Processing dataset from: {dataset_path}")
    print(f"Extracting {subset_percentage}% of the data...")
    
    output_dir = os.path.join(os.path.dirname(dataset_path), "bdd100k_subset_yolo")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    
    # BDD100K standard classes mapping
    class_mapping = {
        "pedestrian": 0,
        "rider": 1,
        "car": 2,
        "truck": 3,
        "bus": 4,
        "train": 5,
        "motorcycle": 6,
        "bicycle": 7,
        "traffic light": 8,
        "traffic sign": 9
    }
    
    # Write classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cls_name, cls_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{cls_name}\n")
    
    # Find JSON annotation file (assuming BDD100k format for this example)
    train_json = os.path.join(dataset_path, "train", "annotations", "bdd100k_labels_images_train.json")
    train_img_dir = os.path.join(dataset_path, "train", "images")
    
    # Fallbacks if the specific subfolder structure is different
    if not os.path.exists(train_json):
        # Searching globally for JSON in the dataset path
        json_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        if json_files:
            train_json = json_files[0]
            print(f"Using JSON file found at: {train_json}")
        else:
            print("No JSON label file found. Assuming YOLO format is already present or dataset is empty.")
            return output_dir

    print(f"Loading annotations from {train_json}...")
    with open(train_json, 'r') as f:
        data = json.load(f)
        
    # Data might be a list of dictionaries (BDD100k format)
    if not isinstance(data, list):
        print("Unsupported JSON format. Expected a list of image annotations.")
        return output_dir
        
    # Sample subset
    sample_size = max(1, int(len(data) * (subset_percentage / 100.0)))
    sampled_data = random.sample(data, sample_size)
    print(f"Sampled {sample_size} images out of {len(data)}")
    
    # Process images and labels
    processed_count = 0
    for item in sampled_data:
        img_name = item.get("name")
        labels = item.get("labels", [])
        
        # Locate the image
        img_src_path = None
        for root, _, files in os.walk(dataset_path):
            if img_name in files:
                img_src_path = os.path.join(root, img_name)
                break
                
        if not img_src_path:
            continue
            
        # Copy image
        img_dest_path = os.path.join(output_dir, 'images', 'train', img_name)
        shutil.copy2(img_src_path, img_dest_path)
        
        # We need image dimensions for YOLO normalization
        # In BDD100k, default size is usually 1280x720, but it's safer to read if possible.
        # Assuming standard BDD100K 1280x720 for efficiency without loading cv2 inside component.
        img_w, img_h = 1280.0, 720.0
        
        # Create YOLO label file
        label_file_name = os.path.splitext(img_name)[0] + ".txt"
        label_dest_path = os.path.join(output_dir, 'labels', 'train', label_file_name)
        
        with open(label_dest_path, 'w') as f_out:
            for label in labels:
                category = label.get("category")
                if category not in class_mapping:
                    continue
                    
                cls_id = class_mapping[category]
                
                box2d = label.get("box2d")
                if not box2d:
                    continue
                    
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])
                
                # Convert to YOLO format: x_center, y_center, width, height (normalized)
                x_c = ((x1 + x2) / 2.0) / img_w
                y_c = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                # Clip values to [0, 1]
                x_c, y_c, w, h = max(0, min(1, x_c)), max(0, min(1, y_c)), max(0, min(1, w)), max(0, min(1, h))
                
                f_out.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
        processed_count += 1

    print(f"Processed and converted {processed_count} images to YOLO format.")
    return output_dir


# ==========================================
# 2. Uploading Component
# ==========================================
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def data_uploading_step(processed_dataset_path: str, dataset_name: str) -> str:
    from clearml import Dataset
    import os
    
    dataset_project = "MLOps_Level2"
    
    print(f"Creating ClearML Dataset: {dataset_name} in project {dataset_project}")
    dataset = Dataset.create(
        dataset_project=dataset_project,
        dataset_name=dataset_name
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
    name='Extract_Convert_Upload_Pipeline',
    project='MLOps_Level2',
    version='1.0',
    add_pipeline_tags=False,
    # Setting the default queue for pipeline steps
    default_queue="data_engineer"
)
def extract_and_upload_pipeline(dataset_path: str, subset_percentage: int, output_dataset_name: str):
    """
    1. Extract a subset from the large dataset and convert BDD JSON to YOLO.
    2. Upload the processed dataset to ClearML.
    """
    processed_path = data_preprocessing_step(
        dataset_path=dataset_path, 
        subset_percentage=subset_percentage
    )
    
    dataset_id = data_uploading_step(
        processed_dataset_path=processed_path,
        dataset_name=output_dataset_name
    )
    
    return dataset_id

# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == '__main__':
    # Pipeline execution parameters
    input_ds_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data_preprocessing', 'datasets', 'bdd100k'))
    percentage = 1
    ds_name = f"BDD100k_{percentage}percent_YOLO"
    
    print(f"Initiating pipeline with {percentage}% extraction from {input_ds_path}")

    if is_ci:
        # --- CI Mode (GitHub Actions) ---
        # Do NOT call run_locally(). This will submit the pipeline as a ClearML
        # Task to the server. The local data_engineer agent will pick it up and
        # execute it on the local machine where the dataset actually lives.
        print("[*] CI mode: Submitting pipeline to ClearML. Local agent will execute.")
    else:
        # --- Local Debug Mode ---
        # run_locally() makes the pipeline run entirely on this machine.
        # Useful for local testing and debugging.
        print("[*] Local mode: Running pipeline locally for debugging.")
        PipelineDecorator.run_locally()
    
    extract_and_upload_pipeline(
        dataset_path=input_ds_path,
        subset_percentage=percentage,
        output_dataset_name=ds_name
    )
