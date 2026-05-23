import os
import sys

def run_step(cmd):
    print(f"========== Running: {cmd} ==========")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error: Step failed with exit code {ret}")
        sys.exit(ret)

if __name__ == "__main__":
    from clearml import Task
    
    print("Starting CI Pipeline Execution...")
    
    # Defaults
    subset_percentage = 0.1
    yolo_epochs = 1
    gnn_epochs = 1
    hpo_trials = 2
    multi_epochs = 1
    dataset_path = "data_preprocessing/datasets/bdd100k"
    
    # If running within ClearML, get parameters
    task = Task.current_task()
    if task:
        params = task.get_parameters_as_dict()
        general = params.get("General", {})
        
        subset_percentage = float(general.get("subset_percentage", subset_percentage))
        yolo_epochs = int(general.get("yolo_epochs", yolo_epochs))
        gnn_epochs = int(general.get("gnn_epochs", gnn_epochs))
        hpo_trials = int(general.get("hpo_trials", hpo_trials))
        multi_epochs = int(general.get("multi_epochs", multi_epochs))
        dataset_path = str(general.get("dataset_path", dataset_path))
        print("Loaded parameters from ClearML task")
    
    print(f"Parameters: dataset_path={dataset_path}, subset_percentage={subset_percentage}, yolo_epochs={yolo_epochs}, gnn_epochs={gnn_epochs}, hpo_trials={hpo_trials}, multi_epochs={multi_epochs}")
    
    # Detach from the CI Trigger task so that each pipeline creates its own Clean Task
    if "CLEARML_TASK_ID" in os.environ:
        del os.environ["CLEARML_TASK_ID"]

    # 1. Data Processing
    run_step(f"python Product/product_piepline/data_processing_pipeline.py --dataset_path \"{dataset_path}\" --subset_percentage {subset_percentage}")
    
    # 2. YOLO Training & Evaluation
    run_step(f"python Product/product_piepline/yolo_training_evaluation_pipeline.py --epochs {yolo_epochs} --yaml_path data_preprocessing/datasets/bdd100k_subset_yolo/dataset.yaml")
    
    # 3. Feature Engineering
    run_step("python Product/product_piepline/feature_engineering_pipeline.py --data_dir data_preprocessing/datasets/bdd100k_subset_yolo/images/train")
    
    # 4. GNN Training & Evaluation
    run_step(f"python Product/product_piepline/gnn_training_evaluation_pipeline.py --epochs {gnn_epochs}")
    
    # 5. Hyperparameter Tuning
    run_step(f"python Product/product_piepline/model_hyper_parameter_tuning.py --epochs {gnn_epochs}")
    
    # 6. Multi-Model Training and Model Selection
    run_step(f"python Product/product_piepline/multi_model_training_and_model_selection.py --epochs {multi_epochs}")

    print("========== CI Pipeline Completed Successfully ==========")
