import logging
import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttentionBlock(nn.Module):
   
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_b_to_a = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_b1 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        self.norm_b2 = nn.LayerNorm(hidden_dim)

        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        
        key_padding_mask_a = None if mask_a is None else ~mask_a.bool()
        key_padding_mask_b = None if mask_b is None else ~mask_b.bool()

        a_attended, attn_weights_a_to_b = self.attn_a_to_b(
            query=feat_a,
            key=feat_b,
            value=feat_b,
            key_padding_mask=key_padding_mask_b,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        enhanced_a = self.norm_a1(feat_a + a_attended)
        enhanced_a = self.norm_a2(enhanced_a + self.ffn_a(enhanced_a))

        b_attended, attn_weights_b_to_a = self.attn_b_to_a(
            query=feat_b,
            key=feat_a,
            value=feat_a,
            key_padding_mask=key_padding_mask_a,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        enhanced_b = self.norm_b1(feat_b + b_attended)
        enhanced_b = self.norm_b2(enhanced_b + self.ffn_b(enhanced_b))

        if mask_a is not None:
            enhanced_a = enhanced_a * mask_a.unsqueeze(-1).to(enhanced_a.dtype)
        if mask_b is not None:
            enhanced_b = enhanced_b * mask_b.unsqueeze(-1).to(enhanced_b.dtype)

        if return_attention:
            return enhanced_a, enhanced_b, {
                'a_to_b': attn_weights_a_to_b,
                'b_to_a': attn_weights_b_to_a,
            }

        return enhanced_a, enhanced_b, None


class AdaptiveGating(nn.Module):
   

    def __init__(
        self,
        descriptor_dim: int = 30,
        hidden_dim: int = 64,
        num_modalities: int = 3,
    ):
        super().__init__()

        self.num_modalities = num_modalities
        self.gate_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_modalities),
        )

        self.base_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.mix_ratio = nn.Parameter(torch.tensor(0.7))

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_net(descriptors)
        learned_weights = F.softmax(gate_logits / self.temperature.clamp(min=0.1), dim=-1)
        base_weights = F.softmax(self.base_weights, dim=0)
        mix_ratio = torch.sigmoid(self.mix_ratio)
        return mix_ratio * learned_weights + (1.0 - mix_ratio) * base_weights.unsqueeze(0)


class GatedAttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int, gate_hidden_dim: Optional[int] = None):
        super().__init__()
        gate_hidden_dim = gate_hidden_dim or max(hidden_dim // 2, 32)

        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
      
        scores = self.score_net(tokens).squeeze(-1) / math.sqrt(tokens.size(-1))

        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(~mask, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            attn_weights = attn_weights * mask.to(attn_weights.dtype)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        pooled = torch.sum(tokens * attn_weights.unsqueeze(-1), dim=1)
        return pooled, attn_weights


class CrossModalAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        descriptor_dim: int = 30,
        dropout: float = 0.1,
        use_adaptive_gating: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim


        if not use_adaptive_gating:
            raise ValueError(
                "3D-MolFusion requires descriptor-guided adaptive gating."
            )
        self.gating_mode = "descriptor_adaptive"
        self.use_adaptive_gating = True

        self.cross_attn_3d_2d = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_attn_3d_fp = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_attn_2d_fp = CrossAttentionBlock(hidden_dim, num_heads, dropout)

        self.pool_3d = GatedAttentionPooling(hidden_dim)
        self.pool_2d = GatedAttentionPooling(hidden_dim)
        self.pool_fp = GatedAttentionPooling(hidden_dim)

        self.adaptive_gate = AdaptiveGating(
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim // 2,
            num_modalities=3,
        )

        self.intra_weights_3d = nn.Parameter(torch.ones(3) / 3)
        self.intra_weights_2d = nn.Parameter(torch.ones(3) / 3)
        self.intra_weights_fp = nn.Parameter(torch.ones(3) / 3)

        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        logger.info("CrossModalAttention initialized (sequence-level fusion)")
        logger.info(f"  hidden_dim: {hidden_dim}")
        logger.info(f"  num_heads: {num_heads}")
        logger.info(f"  gating_mode: {self.gating_mode}")
        logger.info(f"  use_adaptive_gating (derived): {self.use_adaptive_gating}")
        logger.info("  token pooling: light gated attention pooling")

    def forward(
        self,
        feat_3d: torch.Tensor,
        feat_2d: torch.Tensor,
        feat_fp: torch.Tensor,
        tokens_3d: torch.Tensor,
        tokens_2d: torch.Tensor,
        tokens_fp: torch.Tensor,
        mask_3d: Optional[torch.Tensor] = None,
        mask_2d: Optional[torch.Tensor] = None,
        mask_fp: Optional[torch.Tensor] = None,
        descriptors: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
       
        batch_size = feat_3d.size(0)

        if return_attention:
            tokens_3d_from_2d, tokens_2d_from_3d, attn_3d_2d = self.cross_attn_3d_2d(
                tokens_3d, tokens_2d, mask_3d, mask_2d, return_attention=True
            )
            tokens_3d_from_fp, tokens_fp_from_3d, attn_3d_fp = self.cross_attn_3d_fp(
                tokens_3d, tokens_fp, mask_3d, mask_fp, return_attention=True
            )
            tokens_2d_from_fp, tokens_fp_from_2d, attn_2d_fp = self.cross_attn_2d_fp(
                tokens_2d, tokens_fp, mask_2d, mask_fp, return_attention=True
            )
        else:
            tokens_3d_from_2d, tokens_2d_from_3d, _ = self.cross_attn_3d_2d(
                tokens_3d, tokens_2d, mask_3d, mask_2d, return_attention=False
            )
            tokens_3d_from_fp, tokens_fp_from_3d, _ = self.cross_attn_3d_fp(
                tokens_3d, tokens_fp, mask_3d, mask_fp, return_attention=False
            )
            tokens_2d_from_fp, tokens_fp_from_2d, _ = self.cross_attn_2d_fp(
                tokens_2d, tokens_fp, mask_2d, mask_fp, return_attention=False
            )

        pooled_3d_from_2d, pool_weights_3d_from_2d = self.pool_3d(tokens_3d_from_2d, mask_3d)
        pooled_3d_from_fp, pool_weights_3d_from_fp = self.pool_3d(tokens_3d_from_fp, mask_3d)
        pooled_2d_from_3d, pool_weights_2d_from_3d = self.pool_2d(tokens_2d_from_3d, mask_2d)
        pooled_2d_from_fp, pool_weights_2d_from_fp = self.pool_2d(tokens_2d_from_fp, mask_2d)
        pooled_fp_from_3d, pool_weights_fp_from_3d = self.pool_fp(tokens_fp_from_3d, mask_fp)
        pooled_fp_from_2d, pool_weights_fp_from_2d = self.pool_fp(tokens_fp_from_2d, mask_fp)

        w3d = F.softmax(self.intra_weights_3d, dim=0)
        w2d = F.softmax(self.intra_weights_2d, dim=0)
        wfp = F.softmax(self.intra_weights_fp, dim=0)

        enhanced_3d = w3d[0] * feat_3d + w3d[1] * pooled_3d_from_2d + w3d[2] * pooled_3d_from_fp
        enhanced_2d = w2d[0] * feat_2d + w2d[1] * pooled_2d_from_3d + w2d[2] * pooled_2d_from_fp
        enhanced_fp = wfp[0] * feat_fp + wfp[1] * pooled_fp_from_3d + wfp[2] * pooled_fp_from_2d

        if descriptors is None:
            raise ValueError(
                "Descriptor-guided adaptive gating requires molecular descriptors."
            )
        weights = self.adaptive_gate(descriptors)

        weighted_3d = enhanced_3d * weights[:, 0:1]
        weighted_2d = enhanced_2d * weights[:, 1:2]
        weighted_fp = enhanced_fp * weights[:, 2:3]

        fused = self.final_fusion(torch.cat([weighted_3d, weighted_2d, weighted_fp], dim=-1))

        result = {
            'fused': fused,
            'weights': weights,
            'enhanced_3d': enhanced_3d,
            'enhanced_2d': enhanced_2d,
            'enhanced_fp': enhanced_fp,
            'pooled_3d_from_2d': pooled_3d_from_2d,
            'pooled_3d_from_fp': pooled_3d_from_fp,
            'pooled_2d_from_3d': pooled_2d_from_3d,
            'pooled_2d_from_fp': pooled_2d_from_fp,
            'pooled_fp_from_3d': pooled_fp_from_3d,
            'pooled_fp_from_2d': pooled_fp_from_2d,
            'pooling_weights': {
                '3d_from_2d': pool_weights_3d_from_2d,
                '3d_from_fp': pool_weights_3d_from_fp,
                '2d_from_3d': pool_weights_2d_from_3d,
                '2d_from_fp': pool_weights_2d_from_fp,
                'fp_from_3d': pool_weights_fp_from_3d,
                'fp_from_2d': pool_weights_fp_from_2d,
            },
        }

        if return_attention:
            result['attention_weights'] = {
                '3d_2d': attn_3d_2d,
                '3d_fp': attn_3d_fp,
                '2d_fp': attn_2d_fp,
            }

        return result


def create_cross_modal_attention(config: Dict[str, Any] = None) -> CrossModalAttention:
    
    config = config or {}

    return CrossModalAttention(
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        descriptor_dim=config.get('descriptor_dim', 30),
        dropout=config.get('dropout', 0.1),
        use_adaptive_gating=config.get('use_adaptive_gating', True),
    )
