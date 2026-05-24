import os
import sys
import shutil
import json
from clearml import PipelineDecorator, Dataset

# ==========================================
# 0. Secure Configuration Loading
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 【已调整】脚本在 data_preprocessing/ 下，往上一级就是项目根目录 InfinifyX/
local_config_path = os.path.abspath(os.path.join(current_dir, '..', 'clearml.conf'))

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
# 1. Full Preprocessing Component (YOLO 100%)
# ==========================================
@PipelineDecorator.component(cache=True, execution_queue="data_engineer")
def data_preprocessing_step(dataset_path: str) -> str:
    import os
    import json
    import shutil
    
    print(f"Starting FULL conversion for dataset from: {dataset_path}")
    
    # 期望的输出目录：bdd100k_full_yolo
    output_dir = os.path.join(os.path.dirname(dataset_path), "bdd100k_full_yolo")
    if os.path.exists(output_dir):
        print(f"[*] Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
        
    # BDD100K 标准 10 分类映射
    class_mapping = {
        "pedestrian": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
        "train": 5, "motorcycle": 6, "bicycle": 7, "traffic light": 8, "traffic sign": 9
    }
    
    # 写入 classes.txt
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cls_name, cls_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{cls_name}\n")
            
    # BDD100K 官方默认分辨率
    IMG_W, IMG_H = 1280.0, 720.0
    
    # 同时全量循环处理 train 和 val 划分
    for split in ['train', 'val']:
        print(f"\n--- Processing Split: {split} ---")
        
        split_dir = os.path.join(dataset_path, split)
        ann_dir = os.path.join(split_dir, "annotations")
        
        if not os.path.exists(ann_dir):
            print(f"[WARN] Skipping {split}: Annotation directory {ann_dir} not found.")
            continue
            
        # 寻找目录下的标注 json 文件
        json_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        if not json_files:
            print(f"[WARN] Skipping {split}: No JSON file found in {ann_dir}.")
            continue
            
        json_path = os.path.join(ann_dir, json_files[0])
        print(f"Loading annotations from: {json_path}")
        
        with open(json_path, 'r') as f:
            data_list = json.load(f)
            
        if not isinstance(data_list, list):
            print(f"[ERROR] Expected JSON list for BDD100K raw format, got {type(data_list)}. Skipping {split}.")
            continue
            
        # 创建 YOLO 规范子目录
        dst_img_dir = os.path.join(output_dir, 'images', split)
        dst_lbl_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        processed_count = 0
        
        # 100% 全量遍历
        for item in data_list:
            img_name = item.get("name")
            labels = item.get("labels", [])
            
            if not img_name:
                continue
                
            # 优先在对应 split 的目录下寻找图片
            src_img_path = os.path.join(split_dir, "images", img_name)
            if not os.path.exists(src_img_path):
                src_img_path = os.path.join(split_dir, img_name)
                
            # 如果依然找不到，兜底全局搜索
            if not os.path.exists(src_img_path):
                for root, _, files in os.walk(dataset_path):
                    if img_name in files:
                        src_img_path = os.path.join(root, img_name)
                        break
                        
            if not os.path.exists(src_img_path):
                continue
                
            # 复制图片
            shutil.copy2(src_img_path, os.path.join(dst_img_dir, img_name))
            
            # 创建 YOLO label 文本
            label_file_name = os.path.splitext(img_name)[0] + ".txt"
            label_dest_path = os.path.join(dst_lbl_dir, label_file_name)
            
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
                    
                    # 转换至 YOLO 归一化格式
                    w = x2 - x1
                    h = y2 - y1
                    x_c = (x1 + w / 2.0) / IMG_W
                    y_c = (y1 + h / 2.0) / IMG_H
                    norm_w = w / IMG_W
                    norm_h = h / IMG_H
                    
                    # 边界裁剪保护限制在 [0, 1] 之间
                    x_c, y_c, norm_w, norm_h = (
                        max(0.0, min(1.0, x_c)), max(0.0, min(1.0, y_c)), 
                        max(0.0, min(1.0, norm_w)), max(0.0, min(1.0, norm_h))
                    )
                    
                    f_out.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
            processed_count += 1
            
        print(f"[OK] Successfully processed {processed_count} images for '{split}'.")
        
    return output_dir


# ==========================================
# 2. Uploading Component
# ==========================================
@PipelineDecorator.component(cache=False, execution_queue="data_engineer")
def data_uploading_step(processed_dataset_path: str, dataset_name: str) -> str:
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
    name='Full_Convert_Upload_Pipeline',
    project='MLOps_Level2',
    version='1.0',
    add_pipeline_tags=False,
    default_queue="data_engineer"
)
def extract_and_upload_pipeline(dataset_path: str, output_dataset_name: str, skip_upload: bool = True):
    processed_path = data_preprocessing_step(dataset_path=dataset_path)
    
    if skip_upload:
        print("[skip_upload] Skipping ClearML dataset upload (default behaviour).")
        return processed_path
        
    dataset_id = data_uploading_step(
        processed_dataset_path=processed_path,
        dataset_name=output_dataset_name
    )
    return dataset_id


# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == '__main__':
    from clearml import Task as _Task

    clearml_task_id = os.environ.get('CLEARML_TASK_ID')

    if clearml_task_id:
        # --- Agent 模式 ---
        print(f"[*] Running inside ClearML Agent. Task ID: {clearml_task_id}")
        task = _Task.init(continue_last_task=clearml_task_id)

        params = task.get_parameters()
        ds_path = params.get("General/dataset_path", "data_preprocessing/datasets/bdd100k")
        ds_name = params.get("General/output_dataset_name", "BDD100k_Full_YOLO")
        skip_up = str(params.get("General/skip_upload", "true")).lower() == "true"

        if not os.path.isabs(ds_path):
            local_dataset_path = r"D:\UTS\2026Autumn\42174 Artificial Intelligence Studio\Infinity\InfinifyX\data_preprocessing\datasets\bdd100k"
            if os.path.exists(local_dataset_path):
                ds_path = local_dataset_path
                print(f"[*] Using local dataset at: {ds_path}")
            else:
                ds_path = os.path.abspath(ds_path)
                print(f"[*] Resolved dataset path to: {ds_path}")

        print(f"Initiating FULL pipeline execution using data from {ds_path}")
        PipelineDecorator.run_locally()

        extract_and_upload_pipeline(
            dataset_path=ds_path,
            output_dataset_name=ds_name,
            skip_upload=skip_up
        )
    else:
        # --- 本地调试模式 ---
        # 【已调整】当前目录已经是 data_preprocessing，直接定位到子目录 datasets/bdd100k 即可
        input_ds_path = os.path.abspath(os.path.join(current_dir, 'datasets', 'bdd100k'))
        ds_name = "BDD100k_Full_YOLO"

        print(f"Initiating FULL pipeline locally from {input_ds_path}")
        print("[*] Local mode: Running pipeline locally for debugging.")
        PipelineDecorator.run_locally()

        extract_and_upload_pipeline(
            dataset_path=input_ds_path,
            output_dataset_name=ds_name,
            skip_upload=True  # 本地转换，默认跳过长耗时的远程上传任务
        )