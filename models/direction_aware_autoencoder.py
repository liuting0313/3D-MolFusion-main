import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Conv3DEncoderOnly(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [16, 32, 64, 128]
        
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels[-1]
        
        encoder_layers = []
        prev_ch = in_channels
        
        for out_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv3d(prev_ch, out_ch, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
            ])
            prev_ch = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        return self.encoder(x)



class Directional3DProcessor(nn.Module):
   
    
    GROUP_INDICES = {
        'FR': [0, 3],  
        'BB': [5, 1], 
        'TL': [4, 2], 
    }
    
    def __init__(self, in_channels: int = 128, out_channels: int = 64):
        super().__init__()
        
        self.out_channels = out_channels
        
      
        self.conv_fr = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_bb = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_tl = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, encoded_3d: torch.Tensor) -> torch.Tensor:
       
        fr = torch.stack([encoded_3d[:, :, 0], encoded_3d[:, :, 3]], dim=2)  
        bb = torch.stack([encoded_3d[:, :, 5], encoded_3d[:, :, 1]], dim=2) 
        tl = torch.stack([encoded_3d[:, :, 4], encoded_3d[:, :, 2]], dim=2) 
        
        fr_out = self.conv_fr(fr)  
        bb_out = self.conv_bb(bb) 
        tl_out = self.conv_tl(tl) 
        
        x3 = torch.cat([fr_out, bb_out, tl_out], dim=2)  
        
        return x3


class DirectionAwareConvAutoencoder(nn.Module):
  
    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 128,
        hidden_channels: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        if hidden_channels is None:
            hidden_channels = [16, 32, 64, 128]
        
        self.hidden_channels = hidden_channels
        
        self.global_encoder = Conv3DEncoderOnly(
            in_channels=3,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        
        self.directional_processor = Directional3DProcessor(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1], 
        )
        
        pool_out_dim = hidden_channels[-1]  
        
        self.projection = nn.Sequential(
            nn.Linear(pool_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.view_projection = nn.Sequential(
            nn.Linear(pool_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        
        self.direction_embedding = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        
        logger.info("DirectionAwareConvAutoencoder initialized")
        logger.info(f"  image_size: {image_size}")
        logger.info(f"  latent_dim: {latent_dim}")
        logger.info(f"  hidden_channels: {hidden_channels}")
        logger.info("  stage 1: global Conv3D encoder")
        logger.info("  stage 2: direction-aware feature processor")
        logger.info(
            "  serial path: input[3ch] -> "
            f"global[{hidden_channels[-1]}ch] -> "
            f"directional[{hidden_channels[-1]}ch] -> output"
        )
    
    def _build_global_encoder(self, in_channels, hidden_channels, dropout):
        layers = []
        prev_ch = in_channels
        
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv3d(prev_ch, out_ch, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
            ])
            prev_ch = out_ch
        
        return nn.Sequential(*layers)
    
    def _build_directional_encoder(self, in_channels, hidden_channels, dropout):
        feature_extractor = []
        prev_ch = in_channels
        
        for out_ch in hidden_channels:
            feature_extractor.extend([
                nn.Conv3d(prev_ch, out_ch, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
            ])
            prev_ch = out_ch
        
        feature_extractor = nn.Sequential(*feature_extractor)
        
        directional_processor = Directional3DProcessor(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1] // 2,
        )
        
        return nn.ModuleDict({
            'feature_extractor': feature_extractor,
            'directional_processor': directional_processor,
        })
    
    def _extract_sequential_features(self, x):
        global_feat = self.global_encoder(x)  
        directional_feat = self.directional_processor(global_feat) 
        
        return global_feat, directional_feat
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        
       
        x = images.permute(0, 2, 1, 3, 4) 
        
        global_feat, directional_feat = self._extract_sequential_features(x)
        
        view_pooled = directional_feat.mean(dim=[3, 4]).transpose(1, 2)  
        view_tokens = self.view_projection(view_pooled)

        pooled = directional_feat.mean(dim=[2, 3, 4])  

        z_3d = self.projection(pooled) 

        direction_bias = self.direction_embedding.expand(batch_size, -1)
        z_3d = z_3d + direction_bias
        view_tokens = view_tokens + direction_bias.unsqueeze(1)

        output = {
            'z_3d': z_3d,
            'view_tokens': view_tokens,
            'view_mask': torch.ones(batch_size, view_tokens.size(1), dtype=torch.bool, device=view_tokens.device),
        }
        
        return output
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        output = self.forward(images)
        return output['z_3d']


def create_direction_aware_autoencoder(config: Dict[str, Any]) -> DirectionAwareConvAutoencoder:
   
    return DirectionAwareConvAutoencoder(
        image_size=config.get('image_size', 128),
        latent_dim=config.get('latent_dim', 128),
        hidden_channels=config.get('hidden_channels', [16, 32, 64, 128]),
        dropout=config.get('dropout', 0.1),
    )
