import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import logging

from torch_geometric.nn import NNConv
from torch_geometric.utils import to_dense_batch

logger = logging.getLogger(__name__)


class GaussianRBF(nn.Module):
    def __init__(
        self,
        num_rbf: int = 16,
        cutoff: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        centers = torch.linspace(0, cutoff, num_rbf)
        
        widths = torch.full((num_rbf,), (cutoff / num_rbf) * 0.5)
        
        if learnable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.unsqueeze(-1) 
        
        diff = distances - self.centers 
        rbf_output = torch.exp(-((diff / self.widths.clamp(min=1e-6)) ** 2))
        
        cutoff_mask = (distances.squeeze(-1) < self.cutoff).float()
        rbf_output = rbf_output * cutoff_mask.unsqueeze(-1)
        
        return rbf_output


class FourierAngleEncoding(nn.Module):
    
    def __init__(self, num_basis: int = 8):
        super().__init__()
        self.num_basis = num_basis
        frequencies = torch.arange(1, num_basis + 1, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)
    
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        angles = angles.unsqueeze(-1) 
        
        scaled_angles = angles * self.frequencies 
        sin_part = torch.sin(scaled_angles)
        cos_part = torch.cos(scaled_angles)
        
        encoding = torch.cat([sin_part, cos_part], dim=-1) 
        
        return encoding


class MolecularPositionalEncoding(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int = 160,
        num_rbf: int = 16,
        num_heads: int = 4,
        cutoff: float = 10.0,
        learnable_rbf: bool = True,
        use_angle_encoding: bool = True,
        num_angle_basis: int = 8,
        use_direction_encoding: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.cutoff = cutoff
        self.use_angle_encoding = use_angle_encoding
        self.use_direction_encoding = use_direction_encoding
        
        self.rbf = GaussianRBF(num_rbf, cutoff, learnable_rbf)
        
        total_feat_dim = num_rbf
        
        if use_angle_encoding:
            self.angle_encoder = FourierAngleEncoding(num_angle_basis)
            self.angle_feat_dim = 2 * num_angle_basis
            total_feat_dim += self.angle_feat_dim
        
        if use_direction_encoding:
            self.direction_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, num_rbf // 2),
            )
            total_feat_dim += num_rbf // 2
        
        self.attn_bias_proj = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_heads),
        )
        
        self.node_bias_proj = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        logger.info(f"MolecularPositionalEncoding (Enhanced) initialized: "
                   f"num_rbf={num_rbf}, num_heads={num_heads}, cutoff={cutoff}Å, "
                   f"use_angle={use_angle_encoding}, use_direction={use_direction_encoding}")
    
    def compute_pairwise_distances(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
      
        dense_pos, node_mask = to_dense_batch(pos, batch, max_num_nodes=max_nodes)
        diff = dense_pos.unsqueeze(2) - dense_pos.unsqueeze(1)  # [B, max_nodes, max_nodes, 3]
        distances = torch.norm(diff, dim=-1)  # [B, max_nodes, max_nodes]
        
        mask_2d = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # [B, max_nodes, max_nodes]
        
        distances = distances.masked_fill(~mask_2d, 0.0)
        
        return distances, mask_2d
    
    def compute_pairwise_angles(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int,
    ) -> torch.Tensor:
      
        dense_pos, node_mask = to_dense_batch(pos, batch, max_num_nodes=max_nodes)
       
        
        B, N, _ = dense_pos.shape
        
        mask_f = node_mask.unsqueeze(-1).float()  
        centroid = (dense_pos * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
     
        vec_to_atoms = dense_pos - centroid 
        
        vec_norm = vec_to_atoms / (vec_to_atoms.norm(dim=-1, keepdim=True) + 1e-8)  
        
        cos_angles = torch.bmm(vec_norm, vec_norm.transpose(1, 2))
        cos_angles = cos_angles.clamp(-1.0, 1.0) 
        
        angles = torch.acos(cos_angles)  
        

        mask_2d = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        angles = angles.masked_fill(~mask_2d, 0.0)
        
        return angles
    
    def compute_direction_vectors(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int,
    ) -> torch.Tensor:
        dense_pos, node_mask = to_dense_batch(pos, batch, max_num_nodes=max_nodes)
        
        diff = dense_pos.unsqueeze(2) - dense_pos.unsqueeze(1) 
        
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8) 
        directions = diff / dist  
        
        mask_2d = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        directions = directions.masked_fill(~mask_2d.unsqueeze(-1), 0.0)
        
        return directions
    
    def forward(
        self,
        pos: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
        max_nodes: int,
        return_node_bias: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if pos is None or batch is None:
            return None, None
        
        distances, mask = self.compute_pairwise_distances(pos, batch, max_nodes)
        
        rbf_features = self.rbf(distances)  
        
        all_features = [rbf_features]
        
        if self.use_angle_encoding:
            angles = self.compute_pairwise_angles(pos, batch, max_nodes) 
            angle_features = self.angle_encoder(angles) 
            all_features.append(angle_features)
        
        if self.use_direction_encoding:
            directions = self.compute_direction_vectors(pos, batch, max_nodes)  
            direction_features = self.direction_encoder(directions)  
            all_features.append(direction_features)
        
        combined_features = torch.cat(all_features, dim=-1)  
        
        attn_bias = self.attn_bias_proj(combined_features)  
        attn_bias = attn_bias.permute(0, 3, 1, 2)  
        
        mask_expanded = mask.unsqueeze(1).expand_as(attn_bias)
        attn_bias = attn_bias.masked_fill(~mask_expanded, -1e4)
        
      
        node_bias = None
        if return_node_bias:
            node_feat = combined_features.mean(dim=2)
            node_bias = self.node_bias_proj(node_feat)  
        
        return attn_bias, node_bias


class GPSEncoderWithMolPE(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 9,
        edge_dim: int = 3,
        hidden_dim: int = 160,
        output_dim: int = 160,
        num_local_layers: int = 3,
        num_global_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        use_mol_pe: bool = True,
        num_rbf: int = 16,
        pe_cutoff: float = 10.0,
        use_angle_encoding: bool = True,
        use_direction_encoding: bool = True,
        num_angle_basis: int = 8,
    ):
        super().__init__()
        
        self.input_dim = int(input_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_local_layers = int(num_local_layers)
        self.num_global_layers = int(num_global_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.ffn_mult = int(ffn_mult)
        self.use_mol_pe = use_mol_pe
        
        self.node_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        
        self.local_convs = nn.ModuleList()
        self.local_norms = nn.ModuleList()
        for _ in range(self.num_local_layers):
            nn_edge = nn.Sequential(
                nn.Linear(self.edge_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim * self.hidden_dim),
            )
            self.local_convs.append(
                NNConv(self.hidden_dim, self.hidden_dim, nn_edge, aggr="mean")
            )
            self.local_norms.append(nn.LayerNorm(self.hidden_dim))
        
        if use_mol_pe:
            self.mol_pe = MolecularPositionalEncoding(
                hidden_dim=hidden_dim,
                num_rbf=num_rbf,
                num_heads=num_heads,
                cutoff=pe_cutoff,
                use_angle_encoding=use_angle_encoding,
                num_angle_basis=num_angle_basis,
                use_direction_encoding=use_direction_encoding,
            )
        
        self.global_layers = nn.ModuleList()
        for _ in range(self.num_global_layers):
            self.global_layers.append(
                TransformerLayerWithBias(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * ffn_mult,
                    dropout=dropout,
                )
            )
        
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.output_dim),
        )
        
        logger.info(f"GPSEncoderWithMolPE (Enhanced) initialized: "
                   f"use_mol_pe={use_mol_pe}, "
                   f"use_angle={use_angle_encoding}, use_direction={use_direction_encoding}")
    
    def forward(
        self,
        data,
        pos: Optional[torch.Tensor] = None,
        pos_batch: Optional[torch.Tensor] = None,
        *,
        return_sequence: bool = False,
        return_both: bool = False,
    ):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        
        if x is None:
            raise ValueError("GPSEncoderWithMolPE: data.x is None")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"GPSEncoderWithMolPE expected node dimension {self.input_dim}; "
                f"received shape {tuple(x.shape)}."
            )
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if edge_attr is None:
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros((num_edges, self.edge_dim), device=x.device, dtype=x.dtype)
        elif edge_attr.dim() != 2 or edge_attr.size(1) != self.edge_dim:
            raise ValueError(
                f"GPSEncoderWithMolPE expected edge dimension {self.edge_dim}; "
                f"received shape {tuple(edge_attr.shape)}."
            )
        
        h = self.node_embedding(x)
        for conv, ln in zip(self.local_convs, self.local_norms):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = ln(h + h_in)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        dense_h, mask = to_dense_batch(h, batch=batch)  
        key_padding_mask = ~mask  
        
        
        attn_bias = None
        if self.use_mol_pe and pos is not None:
            actual_pos_batch = pos_batch if pos_batch is not None else batch
            attn_bias, _ = self.mol_pe(pos, actual_pos_batch, dense_h.size(1))
        
        for layer in self.global_layers:
            dense_h = layer(dense_h, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        
        dense_out = self.out_proj(dense_h)  
        
        mask_f = mask.unsqueeze(-1).to(dense_out.dtype)
        pooled = (dense_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        
        if return_both:
            return pooled, dense_out, mask
        if return_sequence:
            return dense_out
        return pooled


class TransformerLayerWithBias(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        
        residual = x
        x = self.norm1(x)
        
        Q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N, N]
        
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_scores = attn_scores.masked_fill(mask, -1e4)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # [B, H, N, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        x = residual + attn_output
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


def create_gps_encoder_with_mol_pe(config: dict):
    return GPSEncoderWithMolPE(
        input_dim=config.get('input_dim', 9),
        edge_dim=config.get('edge_dim', 3),
        hidden_dim=config.get('hidden_dim', 160),
        output_dim=config.get('output_dim', 160),
        num_local_layers=config.get('num_local_layers', 3),
        num_global_layers=config.get('num_global_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        ffn_mult=config.get('ffn_mult', 4),
        use_mol_pe=config.get('use_mol_pe', True),
        num_rbf=config.get('num_rbf', 16),
        pe_cutoff=config.get('pe_cutoff', 10.0),
        use_angle_encoding=config.get('use_angle_encoding', True),
        use_direction_encoding=config.get('use_direction_encoding', True),
        num_angle_basis=config.get('num_angle_basis', 8),
    )

