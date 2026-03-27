import os
from clearml import PipelineDecorator, Dataset

# 1. Setup credentials explicitly inside the script
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
os.environ['CLEARML_API_ACCESS_KEY'] = 'R58U0GS1V7DPMV9POEA3L3E6WHH8EV'
os.environ['CLEARML_API_SECRET_KEY'] = 'XQWoX03bgFcB4eJa6Ux8Kt7zmkaVmVdjg-3xBMa1kFdAdpywfbwfAKb8UzS-Rc2WWXU'

# ==========================================
# Step 1: Data Collection Component
# ==========================================
@PipelineDecorator.component(return_values=['dataset_path'], cache=True)
def get_data_step(dataset_name='bdd100k'):
    ds = Dataset.get(dataset_project='InfinifyX', dataset_name=dataset_name)
    local_path = ds.get_local_copy()
    print(f"Data Collection: Dataset '{dataset_name}' ready at {local_path}")
    return local_path

# ==========================================
# Step 2: Data Preprocessing Component
# ==========================================
@PipelineDecorator.component(return_values=['processed_path'], cache=True)
def preprocess_step(raw_data_path):
    print(f"Preprocessing: Processing data at {raw_data_path}...")
    # Add your logic here
    processed_path = raw_data_path 
    return processed_path

# ==========================================
# Step 3: Model Training Component
# ==========================================
@PipelineDecorator.component(return_values=['model_id'], cache=False)
def train_model_step(data_path, epochs=50):
    print(f"Training: Starting YOLO training with data at {data_path}")
    return "final_model_id_001"

# ==========================================
# Step 4: Pipeline Logic Definition
# ==========================================
@PipelineDecorator.pipeline(
    name='InfinifyX_Full_MLOps_Pipeline', 
    project='InfinifyX', 
    version='1.0.0',
    pipeline_execution_queue=None
)
def run_full_mlops_pipeline(dataset_to_use='bdd100k'):
    raw_path = get_data_step(dataset_name=dataset_to_use)
    clean_path = preprocess_step(raw_data_path=raw_path)
    train_model_step(data_path=clean_path, epochs=50)

if __name__ == '__main__':
    run_full_mlops_pipeline(dataset_to_use='bdd100k')
