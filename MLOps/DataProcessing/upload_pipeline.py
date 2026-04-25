import os
import argparse
from clearml import PipelineController


os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml/'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
os.environ['CLEARML_API_ACCESS_KEY'] = 'R58U0GS1V7DPMV9POEA3L3E6WHH8EV'
os.environ['CLEARML_API_SECRET_KEY'] = 'XQWoX03bgFcB4eJa6Ux8Kt7zmkaVmVdjg-3xBMa1kFdAdpywfbwfAKb8UzS-Rc2WWXU'

# 1. Define the specific step (function) for dataset upload
def upload_dataset_step(input_path: str, dataset_name: str, dataset_project: str):
    from clearml import Dataset
    import os
    
    print(f"Creating dataset '{dataset_name}' under project '{dataset_project}'...")
    
    # Check if the path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Specified path not found: {input_path}")

    dataset = Dataset.create(
        dataset_project=dataset_project, 
        dataset_name=dataset_name
    )

    print(f"Adding files from path: {input_path}...")
    dataset.add_files(path=input_path)

    print("Starting upload to ClearML file server...")
    dataset.upload()
    dataset.finalize()

    print(f"Dataset upload and finalization complete. Dataset ID: {dataset.id}")
    return dataset.id

# 2. Main execution block for terminal running
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Upload dataset to ClearML via Pipeline")
    parser.add_argument('--input_path', type=str, default='bdd100k_tiny', help='Local path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='bdd100k_tiny_dataset', help='Name of the dataset in ClearML')
    args = parser.parse_args()

    # Initialize the Pipeline Controller
    pipe = PipelineController(
        name="Dataset_Upload_Pipeline",
        project="InfinityX",
        version="1.0.0",
        add_pipeline_tags=False
    )

    # Add parameters dynamically from command line arguments
    pipe.add_parameter(name="data_input_path", default=args.input_path)
    pipe.add_parameter(name="output_dataset_name", default=args.dataset_name)

    # Add the function step
    pipe.add_function_step(
        name="upload_data_node",
        function=upload_dataset_step,
        function_kwargs=dict(
            input_path="${pipeline.data_input_path}",
            dataset_name="${pipeline.output_dataset_name}",
            dataset_project="InfinityX"
        ),
        function_return=["dataset_id"],
        cache_executed_step=False
    )

    # Execute the Pipeline locally
    print("Starting Pipeline execution...")
    pipe.start_locally(run_pipeline_steps_locally=True)
    print("Pipeline execution finished!")