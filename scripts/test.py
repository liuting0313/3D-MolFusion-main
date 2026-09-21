import os
import sys
import yaml
import argparse
import logging
import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
import traceback

# --- Project root setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Logging setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[console_handler])
logger = logging.getLogger(__name__)

# --- Imports with diagnostics ---
MODULES_AVAILABLE = True
try:
    # Import the model factory and shared training helpers.
    from models import create_model
    from data.dataloader import MoleculeDataset, custom_collate_fn_for_molecules, get_dataset_info
    from utils.metrics import calculate_metrics
    from scripts.train import setup_dataloaders, setup_model, set_seed, NpEncoder
    logger.info("Successfully imported the model and evaluation modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}", exc_info=True)
    MODULES_AVAILABLE = False
except Exception as e_gen:
    logger.error(f"Unexpected error while importing modules: {e_gen}", exc_info=True)
    MODULES_AVAILABLE = False

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained GHMF model on a test set.')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to the experiment output directory containing config.yaml and model checkpoints.')
    parser.add_argument('--checkpoint_name', type=str, default='model_best.pth.tar',
                        help='Name of the checkpoint file to load from the "checkpoints" subdirectory (default: model_best.pth.tar).')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on (default: test).')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda", "cpu"). If None, uses config or autodetects.')
    parser.add_argument('--batch_size_eval', type=int, default=None,
                        help='Batch size for evaluation. Overrides config if set.')
    parser.add_argument('--output_filename', type=str, default=None,
                        help='Filename for the JSON output of test results. Defaults to test_results_{split}_{timestamp}.json.')
    return parser.parse_args()

def load_config_and_checkpoint(experiment_dir: str, checkpoint_name: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Load the saved configuration and checkpoint from an experiment directory."""
    config_path = os.path.join(experiment_dir, 'config.yaml')
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', checkpoint_name)

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return None, None, None
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None, None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration {config_path}: {e}", exc_info=True)
        return None, None, None

    try:
        # Load on CPU first to avoid device-specific checkpoint errors.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_state_dict = checkpoint.get('state_dict')
        # dataset_info is saved in the checkpoint during training.
        dataset_info_from_ckpt = checkpoint.get('dataset_info')

        if model_state_dict is None:
            logger.error(f"Checkpoint {checkpoint_path} does not contain 'state_dict'.")
            return config, None, dataset_info_from_ckpt
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return config, model_state_dict, dataset_info_from_ckpt
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}", exc_info=True)
        return config, None, None

def run_evaluation(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                   device: torch.device, config: Dict, dataset_info: Dict, split_name: str) -> Optional[Dict[str, Any]]:
    """Evaluate a model on one dataset split."""
    model.eval()
    all_preds, all_labels = [], []
    
    # --- Diagnostics counters (help explain val/test gaps) ---
    batches_total = 0
    batches_skipped_none = 0
    batches_skipped_no_labels = 0
    batches_failed = 0

    samples_total = 0
    samples_with_3d = 0
    samples_without_3d = 0
    
    task_type = dataset_info.get('task_type', config.get('model', {}).get('predictor', {}).get('task_type'))
    if not task_type:
        logger.error("Unable to determine task_type; evaluation aborted.")
        return None

    logger.info(f"Evaluating the '{split_name}' split ({len(loader)} batches)...")
    progress_bar_desc = f"Evaluating {split_name}"
    
    # Use a progress bar when tqdm is available.
    try:
        from tqdm import tqdm
        loader_iterable = tqdm(loader, desc=progress_bar_desc, leave=False, disable=config.get('tqdm_disable', False))
    except ImportError:
        logger.info("tqdm is unavailable; evaluation will run without a progress bar.")
        loader_iterable = loader


    with torch.no_grad():
        for batch_idx, batch in enumerate(loader_iterable):
            batches_total += 1
            if batch is None:
                logger.warning(f"Evaluation on '{split_name}': skipping empty batch at index {batch_idx}.")
                batches_skipped_none += 1
                continue
            try:
                batch_on_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_on_device[key] = value.to(device, non_blocking=True)
                    # Handle PyG and other objects exposing a to() method.
                    elif hasattr(value, 'to') and callable(getattr(value, 'to')):
                         batch_on_device[key] = value.to(device)
                    else:
                        batch_on_device[key] = value
                
                labels_tensor = batch_on_device.get('labels')
                if labels_tensor is None:
                    logger.warning(f"Batch {batch_idx} in '{split_name}' has no 'labels'; skipping it.")
                    batches_skipped_no_labels += 1
                    continue

                # Count samples and whether 3D info is available for this batch.
                # For GHMF-style batches, 3D is typically provided via (pos, z, pos_batch) or (rendered_images).
                bs = int(labels_tensor.shape[0]) if hasattr(labels_tensor, "shape") else 0
                samples_total += bs
                has_3d = (
                    (batch_on_device.get('rendered_images') is not None) or
                    (
                        batch_on_device.get('pos') is not None and
                        batch_on_device.get('z') is not None and
                        batch_on_device.get('pos_batch') is not None
                    )
                )
                if has_3d:
                    samples_with_3d += bs
                else:
                    samples_without_3d += bs

                # Test-time augmentation and randomized SMILES are intentionally disabled.
                logits = model(batch_on_device)['logits']
                
                all_preds.append(logits.detach().cpu())
                all_labels.append(labels_tensor.detach().cpu())

            except Exception as e_eval:
                logger.error(f"Evaluation failed for batch {batch_idx} in '{split_name}': {e_eval}", exc_info=True)
                batches_failed += 1
                continue
    
    if not all_labels or not all_preds:
        logger.warning(f"The '{split_name}' split produced no labels or predictions; metrics cannot be computed.")
        # Emit diagnostics to simplify troubleshooting.
        logger.warning(
            f"[Diag] split='{split_name}': batches_total={batches_total}, "
            f"skipped_none={batches_skipped_none}, skipped_no_labels={batches_skipped_no_labels}, failed={batches_failed}"
        )
        logger.warning(
            f"[Diag] split='{split_name}': samples_total={samples_total}, "
            f"with_3d={samples_with_3d}, without_3d={samples_without_3d}"
        )
        return None

    # --- Diagnostics summary ---
    try:
        logger.info(
            f"[Diag] split='{split_name}': batches_total={batches_total}, "
            f"skipped_none={batches_skipped_none}, skipped_no_labels={batches_skipped_no_labels}, failed={batches_failed}"
        )
        if samples_total > 0:
            logger.info(
                f"[Diag] split='{split_name}': samples_total={samples_total}, "
                f"with_3d={samples_with_3d} ({samples_with_3d/samples_total:.1%}), "
                f"without_3d={samples_without_3d} ({samples_without_3d/samples_total:.1%})"
            )
    except Exception:
        pass
    
    try:
        all_labels_tensor_cat = torch.cat(all_labels, dim=0)
        all_preds_tensor_cat = torch.cat(all_preds, dim=0)
        logger.info(f"Completed evaluation on '{split_name}'; samples: {all_labels_tensor_cat.shape[0]}")

        # --- De-normalization logic for regression tasks (similar to train.py) ---
        if task_type == 'regression' and hasattr(loader.dataset, 'normalization_params') and loader.dataset.normalization_params:
            logger.info(f"Performing de-normalization for regression metrics on '{split_name}' split.")
            
            norm_params = loader.dataset.normalization_params
            tasks = dataset_info.get('tasks', []) # Get task names from dataset_info

            if not tasks:
                logger.warning(f"No task names found in dataset_info for de-normalization on '{split_name}'. Skipping.")
            elif len(tasks) != all_labels_tensor_cat.shape[1]:
                logger.warning(f"Mismatch between number of tasks in dataset_info ({len(tasks)}) "
                               f"and label/prediction columns ({all_labels_tensor_cat.shape[1]}) for '{split_name}'. "
                               f"De-normalization might be incorrect or skipped for some tasks.")
            else:
                denorm_labels_list = []
                denorm_preds_list = []
                
                for i, task_name_dn in enumerate(tasks): # Use a different variable name like task_name_dn
                    if task_name_dn in norm_params:
                        mean = norm_params[task_name_dn].get('mean', 0.0)
                        std = norm_params[task_name_dn].get('std', 1.0)
                        
                        if std == 0: # Avoid division by zero if std is exactly 0
                            logger.warning(f"Standard deviation for task '{task_name_dn}' is 0 on '{split_name}'. "
                                           f"Using std=1.0 for de-normalization to avoid errors.")
                            std = 1.0
                        
                        # De-normalize labels and predictions for the current task
                        task_labels_original_scale = all_labels_tensor_cat[:, i] * std + mean
                        task_preds_original_scale = all_preds_tensor_cat[:, i] * std + mean
                        
                        denorm_labels_list.append(task_labels_original_scale.unsqueeze(1))
                        denorm_preds_list.append(task_preds_original_scale.unsqueeze(1))
                        
                        # Optional: Log for the first few tasks/batches if needed for debugging
                        # if batch_idx < 1 and i < 2: 
                        #     logger.debug(f"De-norm for task '{task_name_dn}' on '{split_name}': mean={mean:.4f}, std={std:.4f}")
                    else:
                        logger.warning(f"Normalization parameters for task '{task_name_dn}' not found for '{split_name}'. "
                                       f"Using original (normalized) values for this task's metrics.")
                        denorm_labels_list.append(all_labels_tensor_cat[:, i].unsqueeze(1))
                        denorm_preds_list.append(all_preds_tensor_cat[:, i].unsqueeze(1))

                if denorm_labels_list and denorm_preds_list:
                    all_labels_tensor_cat = torch.cat(denorm_labels_list, dim=1)
                    all_preds_tensor_cat = torch.cat(denorm_preds_list, dim=1)
                    logger.info(f"De-normalization applied to labels and predictions for '{split_name}' metric calculation.")
                else:
                    logger.warning(f"De-normalization lists are empty for '{split_name}'. "
                                   f"Metrics will be calculated on original (potentially normalized) values.")
        # --- End of de-normalization logic ---

    except Exception as e_cat:
        logger.error(f"Failed to concatenate labels/predictions for '{split_name}': {e_cat}", exc_info=True)
        return None

    logger.info(f"Computing metrics for '{split_name}' (task type: {task_type})...")
    metrics = calculate_metrics(all_labels_tensor_cat, all_preds_tensor_cat, task_type)
    logger.info(f"Metrics for '{split_name}': {json.dumps(metrics, cls=NpEncoder, indent=2)}")
    return metrics

def main_test():
    """Run checkpoint evaluation."""
    args = parse_arguments()
    if not MODULES_AVAILABLE:
        logger.fatal("Required modules could not be loaded; evaluation aborted.")
        sys.exit(1)

    logger.info(f"Starting evaluation for experiment directory: {args.experiment_dir}")

    # Load configuration and checkpoint.
    config, model_state_dict, dataset_info_from_ckpt = load_config_and_checkpoint(args.experiment_dir, args.checkpoint_name)
    if config is None or model_state_dict is None:
        logger.fatal("Unable to load the configuration or model state; evaluation aborted.")
        sys.exit(1)

    # Use the training seed for reproducible evaluation.
    set_seed(config['training'].get('seed', 42))

    # Select the evaluation device.
    if args.device:
        device = torch.device(args.device)
    elif 'training' in config and 'device' in config['training']:
        device = torch.device(config['training']['device'])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Optionally override the evaluation batch size.
    if args.batch_size_eval is not None:
        config['data']['batch_size'] = args.batch_size_eval
        logger.info(f"Evaluation batch size overridden to: {args.batch_size_eval}")
    
    # Reconcile checkpoint metadata with the current configuration before model creation.
    if dataset_info_from_ckpt:
        logger.info(f"Dataset metadata loaded from checkpoint: {dataset_info_from_ckpt}")
        # Keep the predictor definition consistent with checkpoint metadata.
        if 'model' in config and 'predictor' in config['model']:
            config['model']['predictor']['task_type'] = dataset_info_from_ckpt.get('task_type', config['model']['predictor'].get('task_type'))
            config['model']['predictor']['num_tasks'] = dataset_info_from_ckpt.get('num_tasks', config['model']['predictor'].get('num_tasks'))
            logger.info(f"Updated predictor configuration from checkpoint metadata: "
                        f"task_type={config['model']['predictor']['task_type']}, "
                        f"num_tasks={config['model']['predictor']['num_tasks']}")
        else:
             logger.warning("model.predictor is missing from the configuration; checkpoint metadata cannot be applied.")


    # Build all loaders and select the requested split.
    try:
        logger.info(f"Preparing the '{args.split}' data loader...")
        all_loaders_info = setup_dataloaders(config, device)
        
        dataset_info_from_setup = all_loaders_info[3] # dataset_info

        target_loader = None
        if args.split == 'train': target_loader = all_loaders_info[0]
        elif args.split == 'val': target_loader = all_loaders_info[1]
        elif args.split == 'test': target_loader = all_loaders_info[2]

        if target_loader is None:
            logger.error(f"Unable to obtain a data loader for the '{args.split}' split.")
            sys.exit(1)
        logger.info(f"Prepared the '{args.split}' data loader ({len(target_loader)} batches).")
        
        # Prefer metadata derived from the currently loaded processed dataset.
        final_dataset_info = dataset_info_from_setup
        if not final_dataset_info.get('task_type') or not final_dataset_info.get('num_tasks'):
            logger.warning("The data loader did not provide valid task_type/num_tasks metadata; falling back to checkpoint metadata.")
            if dataset_info_from_ckpt and dataset_info_from_ckpt.get('task_type') and dataset_info_from_ckpt.get('num_tasks'):
                final_dataset_info = dataset_info_from_ckpt
            else:
                logger.error("Unable to determine valid dataset metadata (task_type and num_tasks); evaluation aborted.")
                sys.exit(1)
        
        logger.info(f"Dataset metadata used for evaluation: {final_dataset_info}")

    except Exception as e_data:
        logger.error(f"Failed to set up data loaders: {e_data}", exc_info=True)
        sys.exit(1)

    # Initialize the model after task metadata has been resolved.
    logger.info("Initializing model...")
    model = setup_model(config, device)
    if model is None:
        logger.fatal("Model initialization failed; evaluation aborted.")
        sys.exit(1)

    # Restore model weights.
    try:
        model.load_state_dict(model_state_dict)
        logger.info("Model weights loaded successfully.")
    except Exception as e_load_state:
        logger.error(f"Failed to load model state_dict: {e_load_state}", exc_info=True)
        # Remove a DataParallel 'module.' prefix when present.
        if all(key.startswith('module.') for key in model_state_dict):
            logger.info("Detected a 'module.' prefix; retrying after removing it...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict)
                logger.info("Model weights loaded successfully after removing the 'module.' prefix.")
            except Exception as e_load_state_no_module:
                 logger.error(f"Failed to load model state_dict after removing the 'module.' prefix: {e_load_state_no_module}", exc_info=True)
                 sys.exit(1)
        else:
            sys.exit(1)
    
    model.to(device)

    # Run evaluation.
    metrics = run_evaluation(model, target_loader, device, config, final_dataset_info, args.split)
    if metrics:
        logger.info(f"Final metrics on '{args.split}': {metrics}")
        # Save results in a shared directory under the project root.
        test_results_base_dir = os.path.join(PROJECT_ROOT, "test_results")
        os.makedirs(test_results_base_dir, exist_ok=True)

        if args.output_filename:
            # Use the user-provided filename directly.
            output_path = os.path.join(test_results_base_dir, args.output_filename)
        else:
            # Generate a dataset- and timestamp-specific filename.
            dataset_name_for_file = config.get('data', {}).get('name', 'unknown_dataset')
            dataset_name_cleaned = "".join(c if c.isalnum() else '_' for c in dataset_name_for_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(test_results_base_dir, f"results_{dataset_name_cleaned}_{args.split}_{timestamp}.json")
        
        try:
            results_to_save = {
                "experiment_dir": args.experiment_dir,
                "checkpoint_name": args.checkpoint_name,
                "evaluation_split": args.split,
                "metrics": metrics,
                "evaluation_timestamp": datetime.now().isoformat(),
                "config_used_for_eval": config
            }
            with open(output_path, 'w') as f:
                json.dump(results_to_save, f, cls=NpEncoder, indent=4)
            logger.info(f"Saved evaluation results to: {output_path}")
        except Exception as e_save:
            logger.error(f"Failed to save evaluation results: {e_save}", exc_info=True)
    else:
        logger.error(f"No evaluation metrics were produced for the '{args.split}' split.")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    if not MODULES_AVAILABLE:
        # Logging may be unavailable if imports fail during startup.
        print("Error: required modules could not be loaded. Check PYTHONPATH and the project layout.", file=sys.stderr)
        sys.exit(1)
    try:
        main_test()
    except Exception as e_main:
        logger.critical(f"Unhandled fatal error in the evaluation entry point: {e_main}", exc_info=True)
        sys.exit(1)
