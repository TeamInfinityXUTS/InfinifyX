import os
import sys
import argparse
from clearml import PipelineController

# ==========================================
# 0. Secure Configuration Loading
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
local_config_path = os.path.join(current_dir, 'clearml.conf')

if os.path.exists(local_config_path):
    os.environ['CLEARML_CONFIG_FILE'] = local_config_path
    print(f"[*] Loaded secure ClearML configuration from: {local_config_path}")
else:
    print(f"[!] Warning: Local clearml.conf not found. Relying on system credentials.")


# ==========================================
# 1. Define the Pipeline Step (Fully Encapsulated)
# ==========================================
def data_augmentation_step(input_dataset_id: str, output_dataset_name: str, output_project: str, augment_multiplier: int) -> str:
    """
    This function acts as a self-contained node in the ClearML Pipeline.
    ALL imports and classes must be defined INSIDE this function for remote execution stability.
    """
    import os
    import glob
    import cv2
    import random
    import albumentations as A
    from clearml import Dataset
    
    # Define the augmentor class inside the step function
    class BDD100KAugmentor:
        def __init__(self, output_base_dir, stage1_p=0.8, stage2_p=0.8):
            self.output_img_dir = os.path.join(output_base_dir, 'images')
            self.output_lab_dir = os.path.join(output_base_dir, 'labels')
            os.makedirs(self.output_img_dir, exist_ok=True)
            os.makedirs(self.output_lab_dir, exist_ok=True)

            self.stage1_transform = A.Compose([
                A.HorizontalFlip(p=0.75),
                A.RandomScale(scale_limit=0.3, p=0.7),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), p=stage1_p)

            self.stage2_transform = A.Compose([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
                A.GaussNoise(std_range=(0.04, 0.2), p=0.3)
            ], p=stage2_p)

        def _read_yolo_labels(self, label_path):
            bboxes, class_labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        data = line.strip().split()
                        if len(data) == 5: 
                            c = int(data[0])
                            x_c, y_c, w, h = [float(x) for x in data[1:]]
                            
                            # Safely clip to [0.0, 1.0]
                            x_min = max(0.0, min(1.0, x_c - w / 2.0))
                            y_min = max(0.0, min(1.0, y_c - h / 2.0))
                            x_max = max(0.0, min(1.0, x_c + w / 2.0))
                            y_max = max(0.0, min(1.0, y_c + h / 2.0))

                            w_new, h_new = x_max - x_min, y_max - y_min
                            if w_new > 0 and h_new > 0:
                                class_labels.append(c)
                                bboxes.append([x_min + w_new / 2.0, y_min + h_new / 2.0, w_new, h_new])
            return bboxes, class_labels

        def process(self, image_path, label_path):
            image = cv2.imread(image_path)
            if image is None: return
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, class_labels = self._read_yolo_labels(label_path)
            if not bboxes: return

            # Stage 1 & 2
            augmented = self.stage1_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            img_aug = self.stage2_transform(image=augmented['image'])['image']
            bboxes_aug = augmented['bboxes']

            base_name = os.path.basename(image_path).split('.')[0]
            new_name = f"{base_name}_aug_{random.randint(1000, 9999)}"
            
            save_img_path = os.path.join(self.output_img_dir, f"{new_name}.jpg")
            cv2.imwrite(save_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
            
            save_lab_path = os.path.join(self.output_lab_dir, f"{new_name}.txt")
            with open(save_lab_path, 'w') as f:
                for i in range(len(bboxes_aug)):
                    f.write(f"{class_labels[i]} " + " ".join([f"{x:.6f}" for x in bboxes_aug[i]]) + "\n")

    augment_multiplier = int(augment_multiplier)
    # --- Execution Logic inside the Step ---
    print(f"Fetching dataset ID: {input_dataset_id}...")
    source_dataset = Dataset.get(dataset_id=input_dataset_id)
    local_path = source_dataset.get_local_copy()
    
    output_dir = os.path.abspath("./pipeline_augmented_temp")
    os.makedirs(output_dir, exist_ok=True)
    augmentor = BDD100KAugmentor(output_base_dir=output_dir)

    image_paths = glob.glob(os.path.join(local_path, "**", "*.jpg"), recursive=True)
    print(f"Starting augmentation for {len(image_paths)} images (Multiplier: {augment_multiplier})...")
    
    for img_path in image_paths:
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        for _ in range(int(augment_multiplier)):
            augmentor.process(img_path, label_path)

    print(f"Creating new dataset '{output_dataset_name}'...")
    new_dataset = Dataset.create(
        dataset_name=output_dataset_name,
        dataset_project=output_project,
        parent_datasets=[input_dataset_id]
    )
    new_dataset.add_files(path=output_dir)
    new_dataset.upload()
    new_dataset.finalize()
    
    print(f"Dataset finalized! ID: {new_dataset.id}")
    return new_dataset.id


# ==========================================
# 2. Main Pipeline Controller Setup
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ClearML Data Augmentation Pipeline Controller")
    parser.add_argument('--input_dataset_id', type=str, required=True, help='Source Dataset ID')
    parser.add_argument('--output_dataset_name', type=str, default='bdd100k_tiny_augmented', help='New Dataset Name')
    parser.add_argument('--augment_multiplier', type=int, default=1, help='Augmentation multiplier')
    args = parser.parse_args()

    # Initialize the Pipeline Controller
    # This will create an entry in the "PIPELINES" UI in ClearML
    pipe = PipelineController(
        name="YOLO_Data_Augmentation_Pipeline",
        project="InfinityX",
        version="1.0.0",
        add_pipeline_tags=False
    )

    # Add pipeline parameters (these show up in the ClearML UI and can be edited before running)
    pipe.add_parameter(name="source_dataset_id", default=args.input_dataset_id)
    pipe.add_parameter(name="target_dataset_name", default=args.output_dataset_name)
    pipe.add_parameter(name="target_project", default="InfinityX/DATASETS/InfinityX")
    pipe.add_parameter(name="multiplier", default=args.augment_multiplier)

    # Add the function as a step in the pipeline graph
    pipe.add_function_step(
        name="augmentation_node",
        function=data_augmentation_step,
        function_kwargs=dict(
            input_dataset_id="${pipeline.source_dataset_id}",
            output_dataset_name="${pipeline.target_dataset_name}",
            output_project="${pipeline.target_project}",
            augment_multiplier="${pipeline.multiplier}"
        ),
        function_return=["new_dataset_id"],
        cache_executed_step=False 
    )

    # Execute the Pipeline
    print("🚀 Starting Pipeline execution...")
    pipe.start_locally(run_pipeline_steps_locally=True)
    print("✅ Pipeline execution finished! Check the PIPELINES tab in ClearML.")