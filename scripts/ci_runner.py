import os
import sys
import subprocess
import json


# ── Path resolution (same logic as end_to_end_pipeline.py) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

KNOWN_WORKSPACES = [
    os.environ.get("INFINIFYX_PROJECT_ROOT", ""),
    PROJECT_ROOT,
    os.getcwd(),
    r"D:\UTS\2026Autumn\42174 Artificial Intelligence Studio\Infinity\InfinifyX",
    "/home/sagemaker-user/InfinifyX",
]


def _find_path(rel_path: str) -> str:
    """Return the first existing absolute path for rel_path across KNOWN_WORKSPACES."""
    for root in KNOWN_WORKSPACES:
        if not root:
            continue
        candidate = os.path.join(root, rel_path)
        if os.path.exists(candidate):
            print(f"  [resolve] {rel_path} -> {candidate}")
            return candidate
    fallback = os.path.join(os.getcwd(), rel_path)
    print(f"  [resolve] {rel_path} -> {fallback}  (NOT FOUND in any workspace)")
    return fallback


def _clean_env():
    """Build a copy of os.environ with all ClearML task / process tracking
    variables removed so that each sub-pipeline creates its own fresh Task."""
    skip_prefixes = ("CLEARML_TASK_ID", "CLEARML_PROC")
    return {k: v for k, v in os.environ.items()
            if not any(k.startswith(p) for p in skip_prefixes)}


def run_step(cmd, env):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, shell=True, env=env, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        print(f"Error: Step failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)


if __name__ == "__main__":
    from clearml import Task
    
    print("="*60)
    print("  InfinifyX CI Pipeline - Sequential Execution")
    print("="*60)
    
    # Defaults
    subset_percentage = 0.1
    yolo_epochs = 1
    gnn_epochs = 1
    hpo_trials = 2
    hpo_epochs = 1
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
        hpo_epochs = int(general.get("hpo_epochs", hpo_epochs))
        multi_epochs = int(general.get("multi_epochs", multi_epochs))
        dataset_path = str(general.get("dataset_path", dataset_path))
        print("[*] Loaded parameters from ClearML task")
    
    # Resolve dataset path across known workspaces
    if not os.path.isabs(dataset_path) or not os.path.exists(dataset_path):
        dataset_path = _find_path(dataset_path)
    
    print(f"\n[*] Parameters:")
    print(f"    dataset_path      = {dataset_path}")
    print(f"    subset_percentage = {subset_percentage}")
    print(f"    yolo_epochs       = {yolo_epochs}")
    print(f"    gnn_epochs        = {gnn_epochs}")
    print(f"    hpo_trials        = {hpo_trials}")
    print(f"    hpo_epochs        = {hpo_epochs}")
    print(f"    multi_epochs      = {multi_epochs}")
    
    # Build a clean environment for child processes
    clean_env = _clean_env()

    # Initialize shared state file
    state_file = os.path.abspath(".ci_state.json")
    with open(state_file, "w") as f:
        json.dump({}, f)

    # 1. Data Processing
    run_step(f'python Product/product_piepline/data_processing_pipeline.py --dataset_path "{dataset_path}" --subset_percentage {subset_percentage} --state_file "{state_file}"', clean_env)
    
    # 2. YOLO Training & Evaluation
    run_step(f'python Product/product_piepline/yolo_training_evaluation_pipeline.py --epochs {yolo_epochs} --state_file "{state_file}"', clean_env)
    
    # 3. Feature Engineering
    run_step(f'python Product/product_piepline/feature_engineering_pipeline.py --state_file "{state_file}"', clean_env)
    
    # 4. GNN Training & Evaluation
    run_step(f'python Product/product_piepline/gnn_training_evaluation_pipeline.py --epochs {gnn_epochs} --state_file "{state_file}"', clean_env)
    
    # 5. Hyperparameter Tuning
    run_step(f'python Product/product_piepline/model_hyper_parameter_tuning.py --epochs {hpo_epochs} --state_file "{state_file}"', clean_env)
    
    # 6. Multi-Model Training and Model Selection
    run_step(f'python Product/product_piepline/multi_model_training_and_model_selection.py --epochs {multi_epochs} --state_file "{state_file}"', clean_env)

    print("\n" + "="*60)
    print("  CI Pipeline Completed Successfully!")
    print("="*60)
