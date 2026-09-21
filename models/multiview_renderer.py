import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw
import io
import logging

logger = logging.getLogger(__name__)

CPK_COLORS = {
    1: (255, 255, 255),  
    6: (128, 128, 128),  
    7: (0, 0, 255),     
    8: (255, 0, 0),       
    9: (0, 255, 0),      
    15: (255, 165, 0),     
    16: (255, 255, 0),   
    17: (0, 255, 0),      
    35: (139, 69, 19),    
    53: (148, 0, 211),   
}
DEFAULT_COLOR = (255, 192, 203)  

VDW_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}
DEFAULT_RADIUS = 1.70


class MultiViewRenderer:
    VIEW_ROTATIONS = {
        'front':  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),      
        'back':   np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),   
        'left':   np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),     
        'right':  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  
        'top':    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),     
        'bottom': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),    
    }
    
    VIEW_ORDER = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    def __init__(
        self,
        image_size: int = 128,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        atom_scale: float = 0.3,
        bond_width: int = 2,
        padding: float = 0.15,
        bond_color: Tuple[int, int, int] = (100, 100, 100),
    ):
       
        self.image_size = image_size
        self.background_color = background_color
        self.atom_scale = atom_scale
        self.bond_width = bond_width
        self.padding = padding
        self.bond_color = bond_color
    
    def render_molecule(
        self,
        coords: np.ndarray,
        atomic_numbers: np.ndarray,
        bonds: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
       
        coords = np.asarray(coords, dtype=np.float32)
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        
        if np.isnan(coords).any() or np.isinf(coords).any():
            logger.warning("Coordinates contain NaN or Inf. Replacing with zeros.")
            coords = np.zeros_like(coords)
            
        coords = coords - coords.mean(axis=0, keepdims=True)
        
        if bonds is None:
            bonds = self._infer_bonds(coords, atomic_numbers)
        
        images = []
        for view_name in self.VIEW_ORDER:
            rotation = self.VIEW_ROTATIONS[view_name]
            rotated_coords = coords @ rotation.T
            
            image = self._render_view(rotated_coords, atomic_numbers, bonds)
            images.append(image)
        
        images = np.stack(images, axis=0)
        images = images.transpose(0, 3, 1, 2)  
        images = images.astype(np.float32) / 255.0  
        
        return images
    
    def _render_view(
        self,
        coords_2d: np.ndarray,
        atomic_numbers: np.ndarray,
        bonds: List[Tuple[int, int]],
    ) -> np.ndarray:
       
        image = Image.new('RGB', (self.image_size, self.image_size), self.background_color)
        draw = ImageDraw.Draw(image)
        
        proj_coords = coords_2d[:, :2] 
        depths = coords_2d[:, 2]  
        
      
        if len(proj_coords) > 0:
            min_xy = proj_coords.min(axis=0)
            max_xy = proj_coords.max(axis=0)
            range_xy = max_xy - min_xy
            max_range = max(range_xy.max(), 1e-6)
            
         
            available_size = self.image_size * (1 - 2 * self.padding)
            scale = available_size / max_range
            
          
            center = (min_xy + max_xy) / 2
            offset = self.image_size / 2
            
           
            img_coords = (proj_coords - center) * scale + offset
            
          
            img_coords = np.clip(img_coords, -1e4, 1e4)
        else:
            img_coords = proj_coords
            scale = 1.0
        
      
        depth_order = np.argsort(-depths)
        
        
        for i, j in bonds:
            if i < len(img_coords) and j < len(img_coords):
                x1, y1 = img_coords[i]
                x2, y2 = img_coords[j]
                draw.line([(x1, y1), (x2, y2)], fill=self.bond_color, width=self.bond_width)
        
        
        for idx in depth_order:
            x, y = img_coords[idx]
            z = atomic_numbers[idx]
            
          
            color = CPK_COLORS.get(int(z), DEFAULT_COLOR)
            radius = VDW_RADII.get(int(z), DEFAULT_RADIUS) * self.atom_scale * scale
            radius = max(radius, 2)  
            
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, fill=color, outline=(50, 50, 50))
        
        return np.array(image)
    
    def _infer_bonds(
        self,
        coords: np.ndarray,
        atomic_numbers: np.ndarray,
        bond_threshold: float = 1.8,
    ) -> List[Tuple[int, int]]:
        n_atoms = len(coords)
        if n_atoms < 2:
            return []
            
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        
        radii = np.array([VDW_RADII.get(int(z), DEFAULT_RADIUS) for z in atomic_numbers])
        threshold_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :]) * 0.6
        
        mask = dist_matrix < threshold_matrix
        mask = np.triu(mask, k=1)
        
        rows, cols = np.nonzero(mask)
        
        bonds = list(zip(rows, cols))
        
        return bonds
    
    def render_batch(
        self,
        coords_list: List[np.ndarray],
        atomic_numbers_list: List[np.ndarray],
        bonds_list: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> np.ndarray:
        batch_images = []
        
        for i in range(len(coords_list)):
            bonds = bonds_list[i] if bonds_list is not None else None
            images = self.render_molecule(
                coords_list[i],
                atomic_numbers_list[i],
                bonds,
            )
            batch_images.append(images)
        
        return np.stack(batch_images, axis=0)


class MultiViewRendererTorch(MultiViewRenderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def render_molecule_torch(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bonds: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        coords_np = coords.detach().cpu().numpy()
        z_np = atomic_numbers.detach().cpu().numpy()
        
        images_np = self.render_molecule(coords_np, z_np, bonds)
        
        return torch.from_numpy(images_np).float()
    
    def render_batch_torch(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
        bonds_list: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        coords_np = coords.detach().cpu().numpy()
        z_np = atomic_numbers.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        
        batch_size = batch_np.max() + 1
        coords_list = []
        z_list = []
        
        for b in range(batch_size):
            mask = batch_np == b
            coords_list.append(coords_np[mask])
            z_list.append(z_np[mask])
        
        images_np = self.render_batch(coords_list, z_list, bonds_list)
        
        return torch.from_numpy(images_np).float()


def create_multiview_renderer(config: Dict[str, Any] = None) -> MultiViewRendererTorch:
    config = config or {}
    
    return MultiViewRendererTorch(
        image_size=config.get('image_size', 128),
        background_color=tuple(config.get('background_color', [255, 255, 255])),
        atom_scale=config.get('atom_scale', 0.3),
        bond_width=config.get('bond_width', 2),
        padding=config.get('padding', 0.15),
        bond_color=tuple(config.get('bond_color', [100, 100, 100])),
    )

