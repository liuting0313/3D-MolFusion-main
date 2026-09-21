
import os
import sys
import yaml
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
import shutil
import json
import traceback
import torch.nn.functional as F
import warnings
from collections import defaultdict
import random
from rdkit import rdBase
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)
logger = logging.getLogger(__name__)
logger.info("Console logging level: INFO. (File logging disabled)")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

logger.info(f"Project root {PROJECT_ROOT} added to sys.path.")

_FocalBCELoss = None

try:
    from utils.losses import FocalBCELoss as _FocalBCELoss_imported
    _FocalBCELoss = _FocalBCELoss_imported
    logger.info("Successfully imported FocalBCELoss from utils.losses.")
except ImportError as e:
    logger.warning(f"FocalBCELoss could not be imported from utils.losses: {e}. Will not be available.")
except Exception as e_gen:
    logger.error(f"An unexpected error occurred while trying to import FocalBCELoss: {e_gen}", exc_info=True)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load` with `weights_only=False`.*")

MODULES_AVAILABLE = True

try:
    from models import create_model, get_model_info
    from data.prepare_datasets import load_and_prepare_dataset
    from data.dataloader import create_ghmf_dataloaders
    from utils.metrics import calculate_metrics
    logger.info("✅ All critical modules imported successfully.")
except ImportError as e:
    logger.critical(f"❌ Failed to import required modules: {e}")
    traceback.print_exc()
    MODULES_AVAILABLE = False
    print("\n!!! CRITICAL ERROR: Not all required modules were imported successfully !!!")
    print("Training cannot proceed. Please fix the import errors above.")
    sys.exit(1)

def setup_model(config: Dict, device: torch.device) -> Any:
    try:
        model_config = config['model'].copy()

        training_config = config.get('training', {})
        if 'loss_weights' in training_config:
            model_config['training'] = {'loss_weights': training_config['loss_weights']}

        model_type = model_config.get('model_type', 'multiview_3dmol')

        model_info = get_model_info(model_type)
        logger.info(f"🚀 Creating {model_info['name']}: {model_info['description']}")
        
        model = create_model(model_config, model_type)
        
        model.to(device)
        logger.info(f"✅ Model moved to device: {device}")
        logger.info(f"📊 Model type: {type(model).__name__}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        if model_info.get('advantages'):
            logger.info("🌟 Model advantages:")
            for advantage in model_info['advantages']:
                logger.info(f"   ✓ {advantage}")
        
        model._model_type = model_type
        model._model_info = model_info
        
        return model

    except Exception as e_err: 
        logger.error(f"An unexpected error occurred during model initialization: {e_err}", exc_info=True)
        sys.exit(1)

def setup_optimizer(config: Dict, model: nn.Module) -> optim.Optimizer:
    optim_config = config.get('training', {}).get('optimizer', {})
    name = optim_config.get('name', 'AdamW').lower()
    lr = float(optim_config.get('lr', 1e-4))
    weight_decay = float(optim_config.get('weight_decay', 0.01))

    betas = optim_config.get('betas', [0.9, 0.999])
    eps = float(optim_config.get('eps', 1e-8))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("The model has no trainable parameters.")

    if name == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = optim.Adam(trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == 'sgd':
        momentum = float(optim_config.get('momentum', 0.9))
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    logger.info(f"Initialized {name.upper()} optimizer")
    return optimizer


def _compute_pos_weight_tensor(config: dict, device: torch.device):
  
    try:
        pw_cfg = config.get('training', {}).get('pos_weight')
        if pw_cfg is None:
            return None

        if isinstance(pw_cfg, (list, tuple)):
            if len(pw_cfg) == 0:
                return None
            return torch.tensor([float(x) for x in pw_cfg], dtype=torch.float32, device=device)

        if isinstance(pw_cfg, str) and pw_cfg.lower() == 'auto':
            data_conf = config.get('data', {})
            dataset_name = data_conf.get('name')
            if not dataset_name:
                logger.warning("pos_weight='auto' requires data.name; skipping automatic class weighting.")
                return None
            processed_dir_cfg = data_conf.get('processed_dir', os.path.join('dataset', 'processed', dataset_name))
            processed_dir_abs = processed_dir_cfg if os.path.isabs(processed_dir_cfg) else os.path.join(PROJECT_ROOT, processed_dir_cfg)
            stats_path = os.path.join(processed_dir_abs, 'split_stats.json')
            if not os.path.exists(stats_path):
                logger.warning(f"pos_weight='auto' requires {stats_path}, but the file does not exist. Using the default weights.")
                return None
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)

                meta = stats.get('_metadata', {}) if isinstance(stats, dict) else {}
                if isinstance(meta, dict) and meta:
                    cached_split_type = meta.get('split_type', None)
                    cached_seed = meta.get('split_seed', None)
                    cached_overlap = meta.get('scaffold_overlap_ratio', None)
                    cached_sha1 = meta.get('indices_sha1', None)
                    if cached_split_type is not None:
                        logger.info(
                            f"🧩 Cached split metadata from split_stats.json: "
                            f"split_type={cached_split_type}, split_seed={cached_seed}, "
                            f"scaffold_overlap_ratio={cached_overlap}, indices_sha1={cached_sha1}"
                        )

                    desired_split_type = data_conf.get('split_type', None)
                    if desired_split_type is not None and cached_split_type is not None:
                        if str(desired_split_type).lower() != str(cached_split_type).lower():
                            logger.warning(
                                f"The current configuration uses data.split_type={desired_split_type}, but the cached data uses split_type={cached_split_type}. "
                                f"Rerun data preprocessing to regenerate *_indices.json and split_stats.json; "
                                f"otherwise, training will continue to use the cached split."
                            )

               
                tasks_in_order = None
                info_path = os.path.join(processed_dir_abs, 'dataset_info.json')
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f_info:
                            ds_info = json.load(f_info)
                        tasks_in_order = ds_info.get('tasks', None)
                    except Exception:
                        pass

                def _is_valid_task_entry(v: dict) -> bool:
                    return isinstance(v, dict) and isinstance(v.get('train', None), dict)

                if isinstance(tasks_in_order, list) and tasks_in_order:
                    task_names = [t for t in tasks_in_order if _is_valid_task_entry(stats.get(t, {}))]
                else:
                    task_names = [k for k, v in stats.items() if (not str(k).startswith('_')) and _is_valid_task_entry(v)]

                if not task_names:
                    logger.warning("pos_weight='auto' found no valid task entries in split_stats.json; the file may be missing or use an incompatible schema.")
                    return None

                weights = []
                for task_name in task_names:
                    task_stat = stats[task_name]
                    pos = float(task_stat.get('train', {}).get('positive', 0))
                    neg = float(task_stat.get('train', {}).get('negative', 0))
                    w = 1.0 if pos <= 0 else round(neg / pos, 4)
                    weights.append(w)

                logger.info(f"Training-set class distribution - positive weights (neg/pos): {dict(zip(task_names, weights))}")
                return torch.tensor(weights, dtype=torch.float32, device=device)
            except Exception as e_read:
                logger.warning(f"Failed to compute pos_weight from {stats_path}: {e_read}")
        return None
    except Exception as e:
        logger.warning(f"Failed to compute the pos_weight tensor: {e}")
        return None


def setup_loss(config: Dict, device: torch.device) -> nn.Module:
    task_type = config['model'].get('task_type', config['model'].get('predictor', {}).get('task_type', 'classification'))
    
    if task_type == 'regression':
        default_loss = config['training'].get('loss', {}).get('main_loss', 'MSELoss')
    else:
        default_loss = 'BCEWithLogitsLoss'
    
    loss_name = config['training'].get('loss_function', default_loss)
    
    logger.info(f"Task type: {task_type}, Using loss function: {loss_name}")
    
    pos_weights = None
    if task_type == 'classification' and loss_name in ['BCEWithLogitsLoss', 'FocalBCELoss', 'FocalBCEWithLogitsLoss']:
        pw_cfg = config['training'].get('pos_weight')
        if pw_cfg is not None:
            pos_weights = _compute_pos_weight_tensor(config, device)
            if pos_weights is not None:
                logger.info(f"Automatically computed pos_weight: {pos_weights.tolist()}")
        
        if pos_weights is None:
            pos_weight_file = config['training'].get('pos_weight_file', None)
            if pos_weight_file and os.path.exists(pos_weight_file):
                try:
                    pos_weights = torch.load(pos_weight_file, map_location=device)
                    logger.info(f"Loaded pos_weight from {pos_weight_file}")
                except Exception as e:
                    pos_weights = None
                    logger.warning(f"Failed to load pos_weight file {pos_weight_file}: {e}")
        
        if pos_weights is None and loss_name == 'BCEWithLogitsLoss':
            logger.info("⚠️  No pos_weight specified. Using default weights (may affect imbalanced datasets).")

    if loss_name == 'BCEWithLogitsLoss':
        task_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    elif loss_name == 'MSELoss':
        task_criterion = nn.MSELoss()
    elif loss_name == 'L1Loss':
        task_criterion = nn.L1Loss()
    elif loss_name == 'SmoothL1Loss':
        task_criterion = nn.SmoothL1Loss()
    elif loss_name == 'CrossEntropyLoss':
        task_criterion = nn.CrossEntropyLoss()
    elif (loss_name == 'FocalBCELoss') and _FocalBCELoss is not None:
        gamma = config['training'].get('focal_gamma', 2.0)
        alpha = config['training'].get('focal_loss_alpha', 0.25)
        task_criterion = _FocalBCELoss(gamma=gamma, alpha=alpha, pos_weight=pos_weights)

        logger.info(f"Initialized {loss_name} with gamma={gamma}, alpha={alpha}")
    elif (loss_name == 'FocalBCEWithLogitsLoss') and _FocalBCELoss is not None: 
        gamma = config['training'].get('focal_gamma', 2.0)
        alpha = config['training'].get('focal_loss_alpha', 0.25)
        task_criterion = _FocalBCELoss(gamma=gamma, alpha=alpha)
        logger.info(f"Initialized {loss_name} (alias of FocalBCELoss) with gamma={gamma}, alpha={alpha}")
    else:
        raise ValueError(f"Unsupported or unimported loss function: {loss_name}")
    
    logger.info(f"Initialized task loss: {loss_name}")
    return task_criterion

def setup_scheduler(config: Dict, optimizer: optim.Optimizer, train_loader_len: Optional[int] = None) -> Optional[optim.lr_scheduler._LRScheduler]:
    scheduler_config = config.get('training', {}).get('scheduler', {})
    if not scheduler_config or not scheduler_config.get('enabled', False):
        logger.info("Learning rate scheduler is disabled")
        return None

    name_lower = scheduler_config.get('name', '').lower()
    
    if name_lower == 'exponentiallr':
        gamma = float(scheduler_config.get('gamma', 0.95))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        logger.info(f"Initialized ExponentialLR scheduler with gamma={gamma}")
        
    elif name_lower == 'steplr':
        step_size = int(scheduler_config.get('step_size', 10))
        gamma = float(scheduler_config.get('gamma', 0.1))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Initialized StepLR scheduler with step_size={step_size}, gamma={gamma}")
        
    elif name_lower == 'multisteplr':
        milestones = scheduler_config.get('milestones', [30, 60, 90])
        gamma = float(scheduler_config.get('gamma', 0.1))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        logger.info(f"Initialized MultiStepLR scheduler with milestones={milestones}, gamma={gamma}")
        
    elif name_lower == 'cosineannealinglr':
        T_max = int(scheduler_config.get('T_max', 50))
        eta_min = float(scheduler_config.get('eta_min', 0))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        logger.info(f"Initialized CosineAnnealingLR scheduler with T_max={T_max}, eta_min={eta_min}")
        
    elif name_lower == 'reducelronplateau':
        mode = scheduler_config.get('mode', 'min')
        factor = float(scheduler_config.get('factor', 0.1))
        patience = int(scheduler_config.get('patience', 10))
        threshold = float(scheduler_config.get('threshold', 1e-4))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                                                       patience=patience, threshold=threshold)
        logger.info(f"Initialized ReduceLROnPlateau scheduler with mode={mode}, factor={factor}, patience={patience}")

    elif name_lower == 'cosineannealingwarmrestarts':
        T_0 = int(scheduler_config.get('T_0', 10))
        T_mult = int(scheduler_config.get('T_mult', 1))
        eta_min = float(scheduler_config.get('eta_min', 0))
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        logger.info(f"Initialized CosineAnnealingWarmRestarts scheduler with T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")

    elif name_lower == 'onecyclelr':
        max_lr = float(scheduler_config.get('max_lr', 0.01))
        epochs_cfg = config.get('training', {}).get('max_epochs', None)
        if epochs_cfg is None:
            epochs_cfg = config.get('training', {}).get('epochs', None)
        total_epochs = int(epochs_cfg) if epochs_cfg is not None else 0
        total_steps = train_loader_len * total_epochs if (train_loader_len and total_epochs > 0) else 1000
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )
        logger.info(f"Initialized OneCycleLR scheduler with max_lr={max_lr}, total_steps={total_steps}")
        
    else:
        scheduler = None
        logger.warning(f"Unsupported scheduler: {name_lower}. No scheduler used.")

    return scheduler

def train_one_epoch(model: Any, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer, 
                    scheduler: Optional[optim.lr_scheduler._LRScheduler],
                    device: torch.device, config: Dict, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Tuple[float, float, float]:
    model.train()
    total_loss_accum, task_loss_accum = 0.0, 0.0
    running_task_loss, running_total_loss = 0.0, 0.0  
    num_batches_processed = 0

    current_epoch_from_config = config.get('current_epoch', '?')
    progress_bar = tqdm(loader, desc=f"Epoch {current_epoch_from_config} Train", leave=False, disable=config.get('tqdm_disable', False))

    grad_clip_setting = config['training'].get('grad_clip', {})
    max_norm_val = None
    if isinstance(grad_clip_setting, dict):
        max_norm_val = grad_clip_setting.get("max_norm")
    elif isinstance(grad_clip_setting, (int, float)):
        max_norm_val = grad_clip_setting

    label_smoothing_factor = config['training'].get('label_smoothing', 0.0)
    task_loss_name = config['training'].get('loss_function', 'BCEWithLogitsLoss') 
    task_type = config.get('model', {}).get('predictor', {}).get('task_type', 'classification') 

   
    sample_weight_cfg = config.get('training', {}).get('sample_weighting', {}) or {}
    sample_weight_enabled = bool(sample_weight_cfg.get('enabled', False)) and (task_type == 'regression')
    sample_weight_strategy = str(sample_weight_cfg.get('strategy', 'threshold')).lower()
    sample_weight_threshold = float(sample_weight_cfg.get('threshold', 0.5))
    sample_weight_high = float(sample_weight_cfg.get('high_weight', 1.0))
    sample_weight_renorm = bool(sample_weight_cfg.get('renormalize', True))

    def _weighted_regression_mse_loss(logits_in: torch.Tensor, labels_in: torch.Tensor) -> torch.Tensor:
        
        labels_f = labels_in.unsqueeze(1).float() if labels_in.dim() == 1 else labels_in.float()
        finite_mask = torch.isfinite(labels_f)
        if not finite_mask.any():
            return logits_in.sum() * 0.0

        labels_safe = torch.where(finite_mask, labels_f, torch.zeros_like(labels_f))

        logits_fp32 = logits_in.float()
        labels_fp32 = labels_safe.float()

        loss_unreduced = F.mse_loss(logits_fp32, labels_fp32, reduction='none')

        weights = torch.ones_like(labels_fp32)
        if sample_weight_strategy == 'threshold' and sample_weight_high != 1.0:
            high_mask = (labels_fp32 >= sample_weight_threshold) & finite_mask
            weights[high_mask] = sample_weight_high

        if sample_weight_renorm:
            w_mean = (weights[finite_mask]).mean().clamp(min=1e-6)
            weights = weights / w_mean

        weighted_loss = loss_unreduced * weights * finite_mask.float()
        return weighted_loss.sum() / finite_mask.sum().clamp(min=1e-6)

    pos_weight_tensor = None
    if task_type == 'classification':
        pw_cfg = config['training'].get('pos_weight')
        if isinstance(pw_cfg, (list, tuple)) and len(pw_cfg) > 0:
            pos_weight_tensor = torch.tensor([float(x) for x in pw_cfg], dtype=torch.float32, device=device)
        elif hasattr(criterion, 'pos_weight') and criterion.pos_weight is not None:
            pos_weight_tensor = criterion.pos_weight.to(device)

    for batch_idx, batch_data in enumerate(progress_bar):
        try:
            optimizer.zero_grad()
            
            if batch_data.get('batch_size', len(batch_data.get('labels', []))) == 0:
                continue
                
            batch_data_to_device = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data_to_device[key] = value.to(device)
                elif hasattr(value, 'to'):  
                    batch_data_to_device[key] = value.to(device)
                elif isinstance(value, dict):
                    batch_data_to_device[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                else:
                    batch_data_to_device[key] = value

            graph_gnn = batch_data_to_device.get('graph_gnn')
            if graph_gnn is None or (hasattr(graph_gnn, 'x') and graph_gnn.x is None):
                logger.warning(f"Batch {batch_idx}: graph_gnn is invalid, skipping")
                continue

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs_dict = model(batch_data_to_device)
                    
                    if isinstance(outputs_dict, dict) and "logits" in outputs_dict:
                        logits = outputs_dict["logits"]
                    else:
                        logits = outputs_dict
                    
                    labels = batch_data_to_device['labels']
                    
                    if hasattr(model, 'module'):
                        actual_model = model.module
                    else:
                        actual_model = model
                    
                    if hasattr(actual_model, 'compute_loss'):
                        loss_dict = actual_model.compute_loss(
                            outputs_dict, labels,
                            pos_weight=pos_weight_tensor,
                            label_smoothing=label_smoothing_factor,
                            criterion=criterion,
                        )
                        task_loss = loss_dict['task_loss']
                        total_loss = loss_dict['total_loss']
                        
                        if sample_weight_enabled:
                            task_loss = _weighted_regression_mse_loss(logits, labels)
                            total_loss = total_loss - loss_dict['task_loss'] + task_loss
                    else:
                        task_loss = criterion(logits, labels)
                        total_loss = task_loss
                    
                    if isinstance(outputs_dict, dict):
                        loss_weights_cfg = config.get('training', {}).get('loss_weights', {}) or {}
                        extra_losses_to_add = {}
                        
                        bridge_w = float(loss_weights_cfg.get('bridge', 0.0) or 0.0)
                        bridge_loss = outputs_dict.get('bridge_loss', None)
                        if bridge_w > 0 and torch.is_tensor(bridge_loss):
                            extra_losses_to_add['bridge_loss'] = bridge_w * bridge_loss
                        
                        align_w = float(loss_weights_cfg.get('align_2d3d', 0.0) or 0.0)
                        if align_w > 0:
                            feat_2d = outputs_dict.get('graph_feat_2d', None)
                            feat_3d = outputs_dict.get('graph_feat_3d', None)
                            if (torch.is_tensor(feat_2d) and torch.is_tensor(feat_3d) and
                                feat_2d.shape == feat_3d.shape and feat_2d.numel() > 0):
                                feat_2d_n = F.normalize(feat_2d, dim=-1)
                                feat_3d_n = F.normalize(feat_3d, dim=-1)
                                align_loss = (1.0 - (feat_2d_n * feat_3d_n).sum(dim=-1)).mean()
                                extra_losses_to_add['align_2d3d'] = align_w * align_loss

                        if extra_losses_to_add:
                            outputs_dict.setdefault('extra_losses', {})
                            outputs_dict['extra_losses'].update(extra_losses_to_add)
                            for loss_name, loss_value in extra_losses_to_add.items():
                                total_loss = total_loss + loss_value
            else:
                outputs_dict = model(batch_data_to_device)
                
                if isinstance(outputs_dict, dict) and "logits" in outputs_dict:
                    logits = outputs_dict["logits"]
                else:
                    logits = outputs_dict
                
                labels = batch_data_to_device['labels']
                
                if hasattr(model, 'module'):
                    actual_model = model.module
                else:
                    actual_model = model
                
                if hasattr(actual_model, 'compute_loss'):
                    loss_dict = actual_model.compute_loss(
                        outputs_dict, labels,
                        pos_weight=pos_weight_tensor,
                        label_smoothing=label_smoothing_factor,
                        criterion=criterion,
                    )
                    task_loss = loss_dict['task_loss']
                    total_loss = loss_dict['total_loss']
                else:
                    task_loss = criterion(logits, labels)
                    total_loss = task_loss
                
            
                if isinstance(outputs_dict, dict):
                    loss_weights_cfg = config.get('training', {}).get('loss_weights', {}) or {}
                    extra_losses_to_add = {}
                    
                    bridge_w = float(loss_weights_cfg.get('bridge', 0.0) or 0.0)
                    bridge_loss = outputs_dict.get('bridge_loss', None)
                    if bridge_w > 0 and torch.is_tensor(bridge_loss):
                        extra_losses_to_add['bridge_loss'] = bridge_w * bridge_loss
                    
                    align_w = float(loss_weights_cfg.get('align_2d3d', 0.0) or 0.0)
                    if align_w > 0:
                        feat_2d = outputs_dict.get('graph_feat_2d', None)
                        feat_3d = outputs_dict.get('graph_feat_3d', None)
                        if (torch.is_tensor(feat_2d) and torch.is_tensor(feat_3d) and
                            feat_2d.shape == feat_3d.shape and feat_2d.numel() > 0):
                            feat_2d_n = F.normalize(feat_2d, dim=-1)
                            feat_3d_n = F.normalize(feat_3d, dim=-1)
                            align_loss = (1.0 - (feat_2d_n * feat_3d_n).sum(dim=-1)).mean()
                            extra_losses_to_add['align_2d3d'] = align_w * align_loss

                    if extra_losses_to_add:
                        outputs_dict.setdefault('extra_losses', {})
                        outputs_dict['extra_losses'].update(extra_losses_to_add)
                        for loss_name, loss_value in extra_losses_to_add.items():
                            total_loss = total_loss + loss_value

            
            if scaler is not None:
                scaler.scale(total_loss).backward()
                
                max_grad_norm = config['training'].get('max_grad_norm', None)
                if max_grad_norm is not None and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                elif 'gradient_clipping' in config['training'] and config['training']['gradient_clipping']['enabled']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping']['max_norm'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                
                max_grad_norm = config['training'].get('max_grad_norm', None)
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                elif 'gradient_clipping' in config['training'] and config['training']['gradient_clipping']['enabled']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping']['max_norm'])
                
                optimizer.step()

            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            current_task_loss = task_loss.item()
            current_total_loss = total_loss.item()
            
            total_loss_accum += current_total_loss
            task_loss_accum += current_task_loss
            num_batches_processed += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            formatted_lr = f"{current_lr:.2E}"  

            progress_bar.set_postfix({
                'loss': f"{current_task_loss:.4f}",
                'lr': formatted_lr
            })

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}", exc_info=True)
            continue

    progress_bar.close()

    avg_total_loss = total_loss_accum / num_batches_processed if num_batches_processed > 0 else float('inf')
    avg_task_loss = task_loss_accum / num_batches_processed if num_batches_processed > 0 else float('inf')

    return avg_total_loss, avg_task_loss, 0.0 

def validate_one_epoch(model: Any, loader: DataLoader, criterion: nn.Module, 
                      device: torch.device, config: Dict) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    num_batches = 0
    task_type = config['model']['predictor']['task_type']
    
    pos_weight_tensor = None
    if task_type == 'classification':
        pw_cfg = config['training'].get('pos_weight')
        if pw_cfg is not None:
            pos_weight_tensor = _compute_pos_weight_tensor(config, device)

    label_smoothing_factor = config['training'].get('label_smoothing', 0.0)
    
    progress_bar = tqdm(loader, desc="Validation", leave=False, disable=config.get('tqdm_disable', False))

    with torch.no_grad():
        for batch_data in progress_bar:
            try:
                if batch_data.get('batch_size', len(batch_data.get('labels', []))) == 0:
                    continue
                    
                batch_data_to_device = {}
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data_to_device[key] = value.to(device)
                    elif hasattr(value, 'to'):  
                        batch_data_to_device[key] = value.to(device)
                    elif isinstance(value, dict):
                        batch_data_to_device[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                    else:
                        batch_data_to_device[key] = value
                
                
                graph_gnn = batch_data_to_device.get('graph_gnn')
                if graph_gnn is None or (hasattr(graph_gnn, 'x') and graph_gnn.x is None):
                    continue
                
                outputs = model(batch_data_to_device)
                logits = outputs["logits"]
                labels = batch_data_to_device['labels']
                
                if hasattr(model, 'compute_loss'):
                    loss_dict = model.compute_loss(
                        outputs, labels,
                        pos_weight=pos_weight_tensor,
                        label_smoothing=label_smoothing_factor,
                        criterion=criterion,
                    )
                    loss = loss_dict['total_loss']
                else:
                    loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                
                all_labels.append(labels.cpu())
                all_predictions.append(logits.cpu())
                
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
            except Exception as e:
                logger.error(f"Error in validation batch: {e}", exc_info=True)
                continue

    progress_bar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    if all_labels and all_predictions:
        all_labels_tensor = torch.cat(all_labels, dim=0)
        all_predictions_tensor = torch.cat(all_predictions, dim=0)

       
        try:
            if task_type == 'regression':
                norm_params = getattr(getattr(loader, 'dataset', None), 'normalization_params', None)
                tasks = getattr(getattr(loader, 'dataset', None), 'tasks', None)

                if norm_params and tasks and len(tasks) == all_labels_tensor.shape[1]:
                    denorm_labels = []
                    denorm_preds = []

                    for i, task_name in enumerate(tasks):
                        p = norm_params.get(task_name, None) if isinstance(norm_params, dict) else None
                        if isinstance(p, dict):
                            mean = float(p.get('mean', 0.0))
                            std = float(p.get('std', 1.0))
                            if std == 0:
                                std = 1.0
                            denorm_labels.append((all_labels_tensor[:, i] * std + mean).unsqueeze(1))
                            denorm_preds.append((all_predictions_tensor[:, i] * std + mean).unsqueeze(1))
                        else:
                            denorm_labels.append(all_labels_tensor[:, i].unsqueeze(1))
                            denorm_preds.append(all_predictions_tensor[:, i].unsqueeze(1))

                    all_labels_tensor = torch.cat(denorm_labels, dim=1)
                    all_predictions_tensor = torch.cat(denorm_preds, dim=1)
                    logger.info("Regression metrics will be computed on de-normalized (original-scale) labels.")
                else:
                    if task_type == 'regression':
                        logger.warning("Regression normalization params not found on dataset; metrics may be on standardized scale.")
        except Exception as e_denorm:
            logger.warning(f"Failed to de-normalize regression labels/predictions for metrics: {e_denorm}")

        metrics = calculate_metrics(all_labels_tensor, all_predictions_tensor, task_type)
    else:
        metrics = {}
        logger.warning("No valid validation batches processed for metrics calculation")
    
    return avg_loss, metrics

def setup_dataloaders(config: Dict, device: torch.device):
    """Setup data loaders for the 3DMolFusion (multiview_3dmol) model."""
    data_config = config['data']
    model_config = config.get('model', {})
    dataset_name = data_config['name']
    data_dir = os.path.join(PROJECT_ROOT, data_config.get('data_dir', 'dataset'))
    batch_size = data_config['batch_size']
    num_workers = data_config.get('num_workers', 0)

    model_type = model_config.get('model_type', 'multiview_3dmol')
    logger.info(f"📊 DataLoader Configuration: model_type={model_type}")
    logger.info(f"🔄 Loading and preparing dataset: {dataset_name} from {data_dir}")
    
    try:
        prepared_data, dataset_info = load_and_prepare_dataset(dataset_name=dataset_name, data_dir=data_dir)
        if not dataset_info or not prepared_data: 
            raise ValueError(f"load_and_prepare_dataset for {dataset_name} returned empty data or info.")
        
        logger.info(f"✅ Dataset '{dataset_name}' loaded. Info: {dataset_info}")
        task_type = dataset_info.get('task_type')
        num_tasks = dataset_info.get('num_tasks')
        if task_type is None or num_tasks is None: 
            raise ValueError("Dataset info must contain 'task_type' and 'num_tasks'.")
        # The factory reads top-level task fields; loss/evaluation read predictor.
        config.setdefault('model', {}).update(task_type=task_type, num_tasks=num_tasks)
        config['model'].setdefault('predictor', {}).update(
            task_type=task_type, num_tasks=num_tasks
        )
        logger.info(f"Updated model task configuration: task_type={task_type}, num_tasks={num_tasks}")
    except FileNotFoundError as e_fnf: logger.error(f"Required processed data or info file not found for {dataset_name}: {e_fnf}"); sys.exit(1)
    except ValueError as e_val: logger.error(f"Error during dataset loading: {e_val}"); sys.exit(1)
    except Exception as e_err: logger.error(f"Error during dataset loading or preparation: {e_err}", exc_info=True); sys.exit(1)

    logger.info("🚀 Using GHMF data loader (graph + fingerprint + conformer + label)")
    train_loader, val_loader, test_loader = create_ghmf_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        required_features=['graph', 'fingerprint', 'conformer', 'label'],
        seed=int(config.get('training', {}).get('seed', 42)),
    )
    if train_loader is None:
        raise RuntimeError("Failed to create the GHMF data loader; training cannot continue.")
    logger.info("✅ GHMF data loaders created successfully")
    return train_loader, val_loader, test_loader, dataset_info

def train(
    config: Dict,
    model: Any,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    output_dir: str,
    dataset_info: Dict,
):
    logger.info("Starting training process...")

    criterion = setup_loss(config, device)
    logger.info(f"Loss function: {type(criterion).__name__}")
    
    optimizer = setup_optimizer(config, model)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    
    scheduler = setup_scheduler(config, optimizer, len(train_loader) if train_loader else None)
    if scheduler:
        logger.info(f"Learning-rate scheduler: {type(scheduler).__name__}")
    
    history = defaultdict(list)
    
    metric_for_best = config['saving']['metric_for_best']
    high_better_metrics = ['val_auc', 'val_roc_auc', 'val_r2', 'val_pearson', 'val_accuracy', 'val_f1']
    if metric_for_best in high_better_metrics:
        best_metric_value = float('-inf')  
    else:
        best_metric_value = float('inf')   
    best_epoch = 0  
    best_model_path = None
    
    early_stop_config = config.get('training', {}).get('early_stopping', {})
    patience = early_stop_config.get('patience', 50)
    min_delta = early_stop_config.get('min_delta', 0.001)
    if 'mode' not in early_stop_config:
        early_stop_config['mode'] = 'max' if 'auc' in early_stop_config.get('monitor', '').lower() else 'min'
    if 'delta' not in early_stop_config:
        early_stop_config['delta'] = min_delta
    epochs_no_improve = 0
    
    logger.info(f"Early stopping config: monitor={early_stop_config.get('monitor', 'val_loss')}, mode={early_stop_config.get('mode')}, patience={patience}")

    mixed_precision_enabled = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = None
    if mixed_precision_enabled:
        scaler = torch.amp.GradScaler(enabled=True) 
        logger.info(f"Mixed precision training enabled with GradScaler for CUDA device.")
    else:
        logger.info("Mixed precision training disabled or not on CUDA device.")

    logger.info(f"Starting training for {config['training']['epochs']} epochs...")

    for epoch_num in range(1, config['training']['epochs'] + 1):
        config['current_epoch'] = epoch_num
        history['epoch'].append(epoch_num)
        steps_this_epoch = len(train_loader) if train_loader else 0

        train_total_loss, train_task_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler if not config['training']['scheduler'].get('update_on_plateau', False) else None,
            device, config, scaler
        )
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_total_loss)
        history['train_task_loss'].append(train_task_loss)
        history['learning_rate'].append(current_lr)

        log_epoch_summary = {"epoch": epoch_num, "train_total_L": train_total_loss, "train_task_L": train_task_loss, "LR": current_lr}

        val_loss, val_metrics = float('nan'), {}
        val_primary_metric_value_epoch = float('nan')

        if val_loader:
            val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device, config)
            history['val_loss'].append(val_loss)
            for k, v in val_metrics.items(): history[f"val_{k}"].append(v)

            log_epoch_summary["val_L"] = val_loss
            for k,v in val_metrics.items(): log_epoch_summary[f"val_{k}"] = v

            monitored_metric_key = early_stop_config['monitor']
            if monitored_metric_key.startswith("val_"):
                base_metric = monitored_metric_key.replace("val_","")
                if base_metric == "auc":
                    base_metric = "roc_auc" 
                val_primary_metric_value_epoch = val_metrics.get(base_metric, val_loss)
            else:
                val_primary_metric_value_epoch = val_metrics.get(monitored_metric_key, val_loss)
            
        else:
            history['val_loss'].append(float('nan'))
            monitored_metric_key = early_stop_config['monitor']
            common_val_metrics_keys = [monitored_metric_key]
            if 'roc_auc' not in common_val_metrics_keys and 'val_roc_auc' not in common_val_metrics_keys : common_val_metrics_keys.append('val_roc_auc')
            for k_common in common_val_metrics_keys:
                 actual_key_common = k_common if k_common.startswith("val_") else f"val_{k_common}"
                 if actual_key_common not in history or len(history[actual_key_common]) < epoch_num:
                     history[actual_key_common].append(float('nan'))
            val_primary_metric_value_epoch = train_total_loss

        formatted_log_items = []
        for k, v in log_epoch_summary.items():
            if k != 'epoch':
                if k == 'LR' and isinstance(v, float):
                    formatted_log_items.append(f"{k}: {v:.2E}")  
                elif isinstance(v, float):
                    formatted_log_items.append(f"{k}: {v:.4f}")
                else:
                    formatted_log_items.append(f"{k}: {v}")
        
        logger.info(f"Epoch {epoch_num}/{config['training']['epochs']} - " + ", ".join(formatted_log_items))

        is_plateau_scheduler = isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
        
        if scheduler and is_plateau_scheduler and config['training']['scheduler'].get('name', '').lower() == 'reducelronplateau':
           
            if val_loader:
                sched_metric = val_primary_metric_value_epoch
                if not np.isnan(sched_metric):
                    scheduler.step(sched_metric)
                else:
                    logger.warning(f"Epoch {epoch_num}: Scheduler metric for ReduceLROnPlateau is NaN. Skipping step.")
            else:
                logger.warning(f"Epoch {epoch_num}: ReduceLROnPlateau specified but no validation loader to get metric.")

        elif scheduler and \
             not isinstance(scheduler, optim.lr_scheduler.OneCycleLR) and \
             not config['training']['scheduler'].get('update_on_step', False) and \
             not is_plateau_scheduler: 
            scheduler.step()
            
        current_metric_for_decision = val_primary_metric_value_epoch if val_loader else train_total_loss

        if np.isnan(current_metric_for_decision):
            logger.warning(f"Metric for early stopping NaN at epoch {epoch_num}. No improvement.")
            epochs_no_improve += 1
            is_best_epoch = False
        else:
            is_best_epoch = False
            early_stopping_mode_val = early_stop_config['mode']
            early_stopping_delta_val = early_stop_config['delta']
            if early_stopping_mode_val == 'max':
                if current_metric_for_decision > best_metric_value + early_stopping_delta_val:
                    best_metric_value, is_best_epoch, epochs_no_improve, best_epoch = current_metric_for_decision, True, 0, epoch_num
                else:
                    epochs_no_improve += 1
            else:
                if current_metric_for_decision < best_metric_value - early_stopping_delta_val:
                    best_metric_value, is_best_epoch, epochs_no_improve, best_epoch = current_metric_for_decision, True, 0, epoch_num
                else:
                    epochs_no_improve += 1

        checkpoint_dir = os.path.join(output_dir, 'checkpoints'); os.makedirs(checkpoint_dir, exist_ok=True)
        current_checkpoint_state = {'epoch': epoch_num, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict() if scheduler else None, 'config': config,
                                    'dataset_info': dataset_info, 'best_metric_value_so_far': best_metric_value}
        
        if is_best_epoch and early_stop_config.get('save_best', True):
            logger.info(f"New best model epoch {epoch_num} with {early_stop_config['monitor']}: {best_metric_value:.4f}")
            
            current_checkpoint_state['monitored_metric_at_best'] = best_metric_value
            
            save_checkpoint(current_checkpoint_state, True, checkpoint_dir)
        
        current_checkpoint_state_original_weights = {'epoch': epoch_num, 
                                                     'state_dict': model.state_dict(), 
                                                     'optimizer': optimizer.state_dict(),
                                                     'scheduler': scheduler.state_dict() if scheduler else None, 
                                                     'config': config,
                                                     'dataset_info': dataset_info, 
                                                     'best_metric_value_so_far': best_metric_value}

        if config['saving'].get('save_every', 0) > 0 and epoch_num % config['saving']['save_every'] == 0:
            save_checkpoint(current_checkpoint_state_original_weights, False, checkpoint_dir, filename=f'checkpoint_epoch_{epoch_num}.pth.tar')
        
        save_checkpoint(current_checkpoint_state_original_weights, False, checkpoint_dir, filename='checkpoint_last.pth.tar')

        if epochs_no_improve >= early_stop_config['patience']: logger.info(f"Early stopping at epoch {epoch_num}."); break

    if np.isinf(best_metric_value) or np.isnan(best_metric_value):
        logger.info(f"Training finished. No valid best {early_stop_config['monitor']} was recorded.")
    else:
        logger.info(f"Training finished. Best {early_stop_config['monitor']}: {best_metric_value:.4f} at epoch {best_epoch if best_epoch > 0 else 'N/A'}")
    summary_data = {"best_monitored_metric": best_metric_value, "best_epoch": best_epoch, "total_epochs_trained": epoch_num, "output_dir": output_dir, "history": history}
    summary_path = os.path.join(output_dir, "training_summary.json")
    try:
        with open(summary_path, 'w') as f: json.dump(summary_data, f, indent=4, cls=NpEncoder)
        logger.info(f"Training summary saved to {summary_path}")
    except Exception as e_sum_save: logger.error(f"Could not save training summary: {e_sum_save}")

    return history, best_metric_value, model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train 3D-MolFusion for molecular property prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to a YAML configuration file')
    parser.add_argument('--seed_override', type=int, default=None, help='Optional random seed that overrides the configuration value')
    args = parser.parse_args()
    return args

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration file: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

def set_seed(seed, deterministic: bool = True, benchmark: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    rdBase.DisableLog('rdApp.warning')  
    rdBase.DisableLog('rdApp.info')     
    logger.info(f"Random seed set to: {seed}")
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_path)
        logger.info(f"Saved the best model to: {best_path}")

def main(cli_args=None, return_results=False):
    if cli_args is None: cli_args = parse_arguments()

    config = load_config(cli_args.config)
    seed_to_use = cli_args.seed_override if cli_args.seed_override is not None else config['training'].get('seed', 42)
    deterministic = bool(config.get('deterministic', True))
    benchmark = bool(config.get('benchmark', False))
    if deterministic and benchmark:
        benchmark = False
    set_seed(seed_to_use, deterministic=deterministic, benchmark=benchmark)
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    ds_name_cleaned = config.get('data', {}).get('name', 'dataset').replace('/', '_')
    exp_name_base = config.get('experiment_name', '3dmolfusion_train')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, config['training'].get('output_dir', 'experiments'), f"{exp_name_base}_{ds_name_cleaned}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Experiment outputs will be saved to: {output_dir}")
    
    try: shutil.copy(cli_args.config, os.path.join(output_dir, 'config.yaml'))
    except Exception as e_cfg_copy: logger.error(f"Failed to copy config to output_dir: {e_cfg_copy}")

    train_loader, val_loader, _, dataset_info = setup_dataloaders(config, device)
    if not dataset_info or 'task_type' not in dataset_info or 'num_tasks' not in dataset_info:
        logger.error(f"Dataset info from {config['data']['name']} is incomplete or missing. Info: {dataset_info}"); sys.exit(1)

    model = setup_model(config, device)

    history, best_metric_val, trained_model = train(
        config,
        model,
        train_loader,
        val_loader,
        device,
        output_dir,
        dataset_info,
    )

    if return_results:
        results = {}
        monitored_metric = config['saving']['metric_for_best']
        results[monitored_metric] = best_metric_val
        best_ckpt_path = os.path.join(output_dir, 'checkpoints', 'model_best.pth.tar')
        if os.path.exists(best_ckpt_path):
            try:
                ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
                results['best_epoch_from_ckpt'] = ckpt.get('epoch')
            except Exception as e_load_final: logger.error(f"Could not load best checkpoint for final results: {e_load_final}")
        logger.info(f"Returning results: {results}")
        return results

if __name__ == "__main__":
    main()
