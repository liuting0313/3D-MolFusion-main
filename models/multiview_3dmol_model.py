import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from .multiview_renderer import MultiViewRendererTorch, create_multiview_renderer
from .direction_aware_autoencoder import create_direction_aware_autoencoder
from .cross_modal_attention import CrossModalAttention, create_cross_modal_attention
from .molecular_positional_encoding import GPSEncoderWithMolPE
from .fingerprint_mlp import FingerprintEncoder

logger = logging.getLogger(__name__)


class MultiView3DMolModel(nn.Module):
   
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.2)
        self.task_type = config.get('task_type', 'classification')
        self.num_tasks = config.get('num_tasks', 1)
        self.node_dim = config.get('node_dim', 9)
        self.edge_dim = config.get('edge_dim', 3)
        self.descriptor_dim = config.get('descriptor_dim', 30)
        self.fingerprint_dim = config.get('fingerprint_dim', 3096)
        

        renderer_config = config.get('multiview_renderer', {})
        self.image_size = renderer_config.get('image_size', 128)
        
        # ============================================================
        # 1. Multi-view renderer (no trainable parameters)
        # ============================================================
        self.renderer = create_multiview_renderer(renderer_config)
        
        # ============================================================
        # 2. Direction-aware 3D encoder
        # ============================================================
        encoder_3d_config = config.get('direction_aware_encoder', {})
        encoder_3d_config['image_size'] = self.image_size
        encoder_3d_config['latent_dim'] = self.hidden_dim
        self.encoder_3d = create_direction_aware_autoencoder(encoder_3d_config)
        
        # ============================================================
        # 3. Geometry-enhanced graph encoder
        # ============================================================
        gps_config = config.get('gps', {})
        self.use_mol_pe = True
        logger.info("Using geometry-enhanced GPSEncoderWithMolPE")
        self.encoder_2d = GPSEncoderWithMolPE(
            input_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=gps_config.get('hidden_dim', self.hidden_dim),
            output_dim=self.hidden_dim,
            num_local_layers=gps_config.get('num_local_layers', 3),
            num_global_layers=gps_config.get('num_global_layers', 2),
            num_heads=gps_config.get('num_heads', self.num_heads),
            dropout=gps_config.get('dropout', self.dropout),
            ffn_mult=gps_config.get('ffn_mult', 4),
            use_mol_pe=True,
            num_rbf=gps_config.get('num_rbf', 16),
            pe_cutoff=gps_config.get('pe_cutoff', 10.0),
            use_angle_encoding=True,
            use_direction_encoding=True,
            num_angle_basis=gps_config.get('num_angle_basis', 8),
        )
        
        # ============================================================
        # 4. Fingerprint encoder
        # ============================================================
        fp_config = config.get('fingerprint_encoder', {})
        self.encoder_fp = FingerprintEncoder(
            input_dim=fp_config.get('input_dim', self.fingerprint_dim),
            hidden_dim=fp_config.get('hidden_dim', 256),
            output_dim=self.hidden_dim,
            num_layers=fp_config.get('num_layers', 2),
            dropout_rate=fp_config.get('dropout_rate', 0.2),
            activation=fp_config.get('activation', 'relu'),
            patch_size=fp_config.get('patch_size', 24),
        )
        
        # ============================================================
        # 5. Cross-modal fusion and descriptor-guided adaptive gating
        # ============================================================
        cross_modal_config = config.get('cross_modal_attention', {})
        cross_modal_config['hidden_dim'] = self.hidden_dim
        cross_modal_config['descriptor_dim'] = self.descriptor_dim
        logger.info("Using CrossModalAttention fusion")
        self.cross_modal_fusion = create_cross_modal_attention(cross_modal_config)
        self.fusion_type = 'attention'
        
        # ============================================================
        # 6. Prediction head
        # ============================================================
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tasks),
        )
        
        self._init_weights()
        self._log_model_info()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _log_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        logger.info("=" * 60)
        logger.info("MultiView3DMolModel initialized:")
        logger.info(f"  hidden_dim: {self.hidden_dim}")
        logger.info(f"  task_type: {self.task_type}")
        logger.info(f"  num_tasks: {self.num_tasks}")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable params: {trainable_params:,}")
        logger.info("=" * 60)

    def _render_molecules(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.renderer.render_batch_torch(coords, atomic_numbers, batch)

    def forward(
        self, data: Dict[str, torch.Tensor], return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        graph = data.get('graph_gnn')
        fingerprints = data.get('fingerprints')
        descriptors = data.get('descriptors')
        rendered_images = data.get('rendered_images') 
        
        pos = data.get('pos')
        z = data.get('z')
        pos_batch = data.get('pos_batch')

        if pos is None and 'conformer_coordinates' in data:
            conformer_coords = data.get('conformer_coordinates')
            conformer_z = data.get('conformer_atomic_numbers')
            conformer_mask = data.get('conformer_attention_mask')

            if conformer_coords is not None and conformer_z is not None:
                pos_list = []
                z_list = []
                batch_list = []

                for i in range(conformer_coords.shape[0]):
                    if conformer_mask is not None:
                        mask = conformer_mask[i].bool()
                        valid_coords = conformer_coords[i][mask]
                        valid_z = conformer_z[i][mask]
                    else:
                        valid_coords = conformer_coords[i]
                        valid_z = conformer_z[i]

                    if valid_coords.numel() > 0:
                        pos_list.append(valid_coords)
                        z_list.append(valid_z)
                        batch_list.append(
                            torch.full((valid_coords.shape[0],), i, dtype=torch.long)
                        )

                if pos_list:
                    pos = torch.cat(pos_list, dim=0)
                    z = torch.cat(z_list, dim=0)
                    pos_batch = torch.cat(batch_list, dim=0)

        if graph is not None:
            batch_size = graph.batch.max().item() + 1
        elif fingerprints is not None:
            batch_size = fingerprints.shape[0]
        else:
            batch_size = 1
        
        # === 1. 3D spatial-visual encoding ===
        if rendered_images is not None:
            images = rendered_images.to(device)
        else:
            images = self._render_molecules(pos, z, pos_batch).to(device)
        encoder_output = self.encoder_3d(images)
        feat_3d = encoder_output['z_3d']
        token_3d = encoder_output['view_tokens']
        mask_3d = encoder_output['view_mask']
        
        # === 2. Geometry-enhanced graph encoding ===
        graph = graph.to(device)
        pos_device = pos.to(device)
        pos_batch_device = pos_batch.to(device) if pos_batch is not None else None
        feat_2d, token_2d, mask_2d = self.encoder_2d(
            graph,
            pos=pos_device,
            pos_batch=pos_batch_device,
            return_both=True,
        )
        
        # === 3. Fingerprint encoding ===
        fingerprints = fingerprints.to(device)
        feat_fp, token_fp, mask_fp = self.encoder_fp(fingerprints, return_both=True)
        
        # === 4. cross-modal fusion ===
        if descriptors is not None:
            descriptors = descriptors.to(device)
        else:
            descriptors = torch.zeros(batch_size, self.descriptor_dim, device=device)
        
        fusion_output = self.cross_modal_fusion(
            feat_3d=feat_3d,
            feat_2d=feat_2d,
            feat_fp=feat_fp,
            tokens_3d=token_3d,
            tokens_2d=token_2d,
            tokens_fp=token_fp,
            mask_3d=mask_3d,
            mask_2d=mask_2d,
            mask_fp=mask_fp,
            descriptors=descriptors,
            return_attention=return_attention,
        )
        
        fused_features = fusion_output['fused']
        weights = fusion_output['weights']
        
        # === 5. predictor ===
        logits = self.predictor(fused_features)
        
        output = {
            'logits': logits,
            'weights': weights,
            'feat_3d': feat_3d,
            'feat_2d': feat_2d,
            'feat_fp': feat_fp,
            'fused_features': fused_features,
        }
        
        if return_attention and 'attention_weights' in fusion_output:
            output['attention_weights'] = fusion_output['attention_weights']
        
        return output
    
    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = output['logits']

        if labels.dim() == 1:
            labels_f = labels.unsqueeze(1).float()
        else:
            labels_f = labels.float()

        finite_mask = torch.isfinite(labels_f)
        if mask is None:
            mask_f = finite_mask
        else:
            mask_f = mask.to(device=labels_f.device, dtype=torch.bool)
            if mask_f.dim() == 1:
                mask_f = mask_f.unsqueeze(1)
            mask_f = mask_f & finite_mask

        if not mask_f.any():
            zero = logits.sum() * 0.0
            return {
                'total_loss': zero,
                'task_loss': zero,
            }

        labels_f = torch.where(mask_f, labels_f, torch.zeros_like(labels_f))

        if label_smoothing is not None and float(label_smoothing) > 0:
            logger.warning_once = getattr(logger, "warning_once", set())
            _msg = f"Ignoring label_smoothing={float(label_smoothing)} to match paper Eq.(21) (weighted BCE only)."
            if _msg not in logger.warning_once:
                logger.warning(_msg)
                logger.warning_once.add(_msg)

        pos_w = None
        if self.task_type == 'classification' and pos_weight is not None:
            if torch.is_tensor(pos_weight):
                pos_w = pos_weight.to(device=logits.device, dtype=logits.dtype)
            else:
                pos_w = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        
        if self.task_type == 'classification':
            if criterion is not None and not isinstance(criterion, nn.BCEWithLogitsLoss):
                labels_for_crit = labels_f.clone()
                labels_for_crit = torch.where(
                    mask_f,
                    labels_for_crit,
                    torch.full_like(labels_for_crit, float('nan')),
                )
                task_loss = criterion(logits, labels_for_crit)
            else:
                if pos_w is None and criterion is not None and hasattr(criterion, 'pos_weight'):
                    crit_pw = getattr(criterion, 'pos_weight', None)
                    if torch.is_tensor(crit_pw):
                        pos_w = crit_pw.to(device=logits.device, dtype=logits.dtype)

                if mask is not None or (not finite_mask.all()):
                    mask_f_dtype = mask_f.to(dtype=logits.dtype)
                    task_loss = F.binary_cross_entropy_with_logits(
                        logits, labels_f, reduction='none', pos_weight=pos_w
                    )
                    task_loss = (task_loss * mask_f_dtype).sum() / mask_f_dtype.sum().clamp(min=1e-6)
                else:
                    task_loss = F.binary_cross_entropy_with_logits(
                        logits, labels_f, pos_weight=pos_w
                    )
        else: 
            if mask is not None or (not finite_mask.all()):
                mask_f_dtype = mask_f.to(dtype=logits.dtype)
                task_loss = F.mse_loss(logits, labels_f, reduction='none')
                task_loss = (task_loss * mask_f_dtype).sum() / mask_f_dtype.sum().clamp(min=1e-6)
            else:
                task_loss = F.mse_loss(logits, labels_f)
        
        total_loss = task_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
        }
    
    
    def get_interpretability(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        with torch.no_grad():
            output = self.forward(data)
        
        return {
            'predictions': output['logits'].cpu().numpy(),
            'weights': output['weights'].cpu().numpy(),
            'weight_names': ['3D_visual', '2D_graph', 'fingerprint'],
        }


def create_multiview_3dmol_model(config: Dict[str, Any]) -> MultiView3DMolModel:
    default_config = {
        'hidden_dim': 128,
        'num_heads': 4,
        'dropout': 0.2,
        'task_type': 'classification',
        'num_tasks': 1,
        'node_dim': 9,
        'edge_dim': 3,
        'descriptor_dim': 30,
        'fingerprint_dim': 3096,
        'multiview_renderer': {
            'image_size': 128,
        },
        'direction_aware_encoder': {
            'hidden_channels': [32, 64, 128, 256],
        },
        'gps': {
            'num_local_layers': 3,
            'num_global_layers': 2,
        },
        'fingerprint_encoder': {
            'hidden_dim': 256,
            'num_layers': 2,
        },
        'cross_modal_attention': {
            'use_adaptive_gating': True,
        },
    }
    
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    final_config = deep_merge(default_config, config)
    
    return MultiView3DMolModel(final_config)

