from clearml import Task
from clearml.automation import PipelineController
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from model_hyper_parameter_tunning_V1.py
PROJECT_NAME = "HPO_Pipeline——GNN_Hyper_Parameter_Tunning"
DEFAULT_GRAPH_CACHE = "graph_cache.pt"
DEFAULT_GRAPH_CACHE_TASK_ID = "b6e2aee4168040729935da12e87d4b1f"
DEFAULT_GRAPH_CACHE_ARTIFACT_NAME = "graph_cache"
EXECUTION_QUEUE = "Yolov8_training_v0.1"


def get_args():
    parser = argparse.ArgumentParser(
        description="Run GNN HPO Pipeline with PipelineController (Multi-Stage)"
    )
    parser.add_argument(
        "--queue",
        default=EXECUTION_QUEUE,
        help="ClearML queue name for remote execution."
    )
    parser.add_argument(
        "--max_epochs_per_trial",
        type=int,
        default=2,
        help="Maximum epochs per trial."
    )
    parser.add_argument(
        "--total_max_jobs",
        type=int,
        default=2,
        help="Total max HPO jobs."
    )
    parser.add_argument(
        "--max_concurrent_tasks",
        type=int,
        default=2,
        help="Max concurrent HPO tasks."
    )
    parser.add_argument(
        "--time_limit_minutes",
        type=int,
        default=20,
        help="Time limit for HPO in minutes."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation set fraction."
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Test set fraction."
    )
    return parser.parse_args()


def run_hpo_pipeline(args):
    """
    Create and run a multi-stage ClearML pipeline for GNN Hyperparameter Optimization.
    
    Pipeline stages:
    1. stage_prepare_graph: Load and prepare graph data
    2. stage_base_training: Train base model with default parameters
    3. stage_hpo: Run hyperparameter optimization
    4. stage_final_model: Train final model with best hyperparameters
    """
    
    # Initialize the pipeline controller
    pipe = PipelineController(
        name="GNN_HPO_Multi_Stage_Pipeline", 
        project=PROJECT_NAME, 
        version="2.0.0", 
        add_pipeline_tags=False
    )

    # Set default execution queue
    pipe.set_default_execution_queue(args.queue)
    logger.info(f"Multi-Stage Pipeline initialized with queue: {args.queue}")

    # Stage 1: Prepare Graph Cache
    # This stage loads the graph data and makes it available as artifact
    pipe.add_step(
        name="stage_prepare_graph",
        base_task_project=PROJECT_NAME,
        base_task_name="GNN_HPO_Base_Task",
        execution_queue=args.queue,
        parameter_override={
            "General/graph_cache": DEFAULT_GRAPH_CACHE,
            "General/graph_cache_task_id": DEFAULT_GRAPH_CACHE_TASK_ID,
            "General/graph_cache_artifact_name": DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
            "General/val_split": args.val_split,
            "General/test_split": args.test_split,
            "General/max_epochs_per_trial": 1,
        }
    )

    # Stage 2: Base Training
    # Train a single model with default parameters to initialize metrics
    pipe.add_step(
        name="stage_base_training",
        parents=["stage_prepare_graph"],
        base_task_project=PROJECT_NAME,
        base_task_name="GNN_HPO_Base_Task",
        execution_queue=args.queue,
        parameter_override={
            "General/graph_cache": DEFAULT_GRAPH_CACHE,
            "General/graph_cache_task_id": "${stage_prepare_graph.id}",
            "General/graph_cache_artifact_name": DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
            "General/hidden_dim": 128,
            "General/dropout": 0.3,
            "General/lr": 5e-4,
            "General/batch_size": 16,
            "General/val_split": args.val_split,
            "General/test_split": args.test_split,
            "General/max_epochs_per_trial": args.max_epochs_per_trial,
        }
    )

    # Stage 3: Hyperparameter Optimization
    # Run HPO to find optimal hyperparameters
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_base_training"],
        base_task_project=PROJECT_NAME,
        base_task_name="HPO: GNN Hyperparameter Optimization",
        execution_queue=args.queue,
        parameter_override={
            "General/graph_cache": DEFAULT_GRAPH_CACHE,
            "General/graph_cache_task_id": DEFAULT_GRAPH_CACHE_TASK_ID,
            "General/graph_cache_artifact_name": DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
            "General/max_epochs_per_trial": args.max_epochs_per_trial,
            "General/total_max_jobs": args.total_max_jobs,
            "General/max_concurrent_tasks": args.max_concurrent_tasks,
            "General/time_limit_minutes": args.time_limit_minutes,
            "General/val_split": args.val_split,
            "General/test_split": args.test_split,
            "General/run_mode": "remote",
            "General/queue": args.queue,
        }
    )

    # Stage 4: Final Model Training
    pipe.add_step(
        name="stage_final_model",
        parents=["stage_hpo"],
        base_task_project=PROJECT_NAME,
        base_task_name="GNN_HPO_Base_Task",
        execution_queue=args.queue,
        parameter_override={
            "General/graph_cache": DEFAULT_GRAPH_CACHE,
            "General/graph_cache_task_id": "${stage_hpo.id}",
            "General/graph_cache_artifact_name": DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
            "General/max_epochs_per_trial": 20,
            "General/val_split": args.val_split,
            "General/test_split": args.test_split,
            "General/hpo_task_id": "${stage_hpo.id}",
        }
    )

    logger.info("\n===== Pipeline Configuration =====")
    logger.info(f"Stage 1: Prepare Graph Cache")
    logger.info(f"Stage 2: Base Training (Epochs: {args.max_epochs_per_trial})")
    logger.info(f"Stage 3: HPO Optimization (Jobs: {args.total_max_jobs}, Concurrent: {args.max_concurrent_tasks})")
    logger.info(f"Stage 4: Final Model Training (Epochs: 100)")
    logger.info(f"Execution Queue: {args.queue}")
    logger.info(f"Time Limit: {args.time_limit_minutes} minutes")
    logger.info("===================================\n")
    
    logger.info(f"Starting multi-stage pipeline...")
    pipe.start_locally()
    
    logger.info("\n===== Pipeline Complete =====")


if __name__ == "__main__":
    args = get_args()
    run_hpo_pipeline(args)
