from clearml import Task
from clearml.automation import PipelineController
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
# Project and queue for ClearML
HPO_PROJECT_NAME = "HPO_Pipeline——GNN_Hyper_Parameter_Tunning"  # HPO orchestration project
BASE_TASK_ID = "74b4a2b8962c43c98f20cac192c1e770"  # Base training task template ID
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
        "--best_hpo_task_id",
        type=str,
        default=None,
        help="Best HPO task ID to use for final model training."
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
    pipe = PipelineController(
        name="GNN_HPO_Multi_Stage_Pipeline", 
        project=HPO_PROJECT_NAME,  # HPO orchestration project
        version="2.0.0", 
        add_pipeline_tags=False
    )

    pipe.set_default_execution_queue(args.queue)
    logger.info(f"Multi-Stage Pipeline initialized")
    logger.info(f"HPO Project: {HPO_PROJECT_NAME}")
    logger.info(f"Base task template ID: {BASE_TASK_ID}")
    logger.info(f"Execution Queue: {args.queue}")

    # Stage 1: Quick Baseline Verification
    pipe.add_step(
        name="stage_quick_baseline",
        base_task_id=BASE_TASK_ID,
        execution_queue=args.queue,
        parameter_override={
            "General/graph_cache": DEFAULT_GRAPH_CACHE,
            "General/graph_cache_task_id": DEFAULT_GRAPH_CACHE_TASK_ID,
            "General/graph_cache_artifact_name": DEFAULT_GRAPH_CACHE_ARTIFACT_NAME,
            "General/hidden_dim": 128,
            "General/dropout": 0.3,
            "General/lr": 5e-4,
            "General/batch_size": 16,
            "General/val_split": args.val_split,
            "General/test_split": args.test_split,
            "General/max_epochs_per_trial": 1,
        }
    )

    # Stage 2: Baseline Training
    pipe.add_step(
        name="stage_baseline_training",
        parents=["stage_quick_baseline"],
        base_task_id=BASE_TASK_ID,
        execution_queue=args.queue,
        parameter_override={
            "General/graph_cache": DEFAULT_GRAPH_CACHE,
            "General/graph_cache_task_id": DEFAULT_GRAPH_CACHE_TASK_ID,
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

    # Stage 3: Hyperparameter Optimization (True HPO Search)
    pipe.add_step(
        name="stage_hpo_search",
        parents=["stage_baseline_training"],
        base_task_id=BASE_TASK_ID,
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

    # Stage 4: Final Model Training (Optional)
    if args.best_hpo_task_id:
        logger.info(f"Using best HPO task {args.best_hpo_task_id} for final model training")
        pipe.add_step(
            name="stage_final_training",
            parents=["stage_hpo_search"],
            base_task_id=args.best_hpo_task_id,
            execution_queue=args.queue,
            parameter_override={
                "General/max_epochs_per_trial": 20,
            }
        )
    else:
        logger.warning("No best_hpo_task_id provided - skipping final model training")
        logger.warning("To enable Stage 4, run after HPO completes with: --best_hpo_task_id <task_id>")

    logger.info("\n" + "="*50)
    logger.info("Pipeline Configuration Summary")
    logger.info("="*50)
    logger.info(f"Stage 1: Quick Baseline Verification (1 epoch)")
    logger.info(f"Stage 2: Baseline Training ({args.max_epochs_per_trial} epochs)")
    logger.info(f"Stage 3: HPO Search (Jobs: {args.total_max_jobs}, Concurrent: {args.max_concurrent_tasks})")
    if args.best_hpo_task_id:
        logger.info(f"Stage 4: Final Training (20 epochs) - Best Task: {args.best_hpo_task_id}")
    logger.info(f"Queue: {args.queue}")
    logger.info(f"Time Limit: {args.time_limit_minutes} minutes")
    logger.info("="*50 + "\n")
    
    logger.info(f"Starting multi-stage pipeline...")
    pipe.start_locally()
    
    logger.info("\n" + "="*50)
    logger.info("Pipeline Complete")
    logger.info("="*50)


if __name__ == "__main__":
    args = get_args()
    run_hpo_pipeline(args)
