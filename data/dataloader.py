import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import math
import logging
import warnings
import functools
import traceback
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch Geometric import failed: {e}. Some functionality may be limited.")
    PYG_AVAILABLE = False
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load` with `weights_only=False.*")
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import random


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'dataset')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
logger.info(f"Dataloader using PROJECT_DIR: {PROJECT_DIR}")
logger.info(f"Dataloader using PROCESSED_DIR: {PROCESSED_DIR}")



class MoleculeDataset(Dataset):
 
    def __init__(
            self,
            samples: List[Dict],          
            dataset_dir: str,              
            task_info: Dict,               
            split: str,                     
            task_name: Optional[str] = None,
            required_features: List[str] = ['smiles', 'graph', 'fingerprint', 'conformer', 'label'],
        ):
     
        self.required_features = required_features
        self.dataset_dir = dataset_dir
        self.samples = samples 
        self.split = split    

        if not self.samples:
            logger.warning(f"MoleculeDataset received an empty list of samples for split '{split}'.")
            self.num_samples = 0
            self.tasks = []
            self.task_type = 'unknown'
            return 

        self.num_samples = len(self.samples)

        
        self.info = task_info
        self.task_type = self.info.get('task_type', 'classification')
        self.tasks = self.info.get('tasks', [])

        
        self.normalization_params = None
        if self.task_type == 'regression':
            norm_file = os.path.join(self.dataset_dir, 'label_normalization.json')
            if os.path.exists(norm_file):
                try:
                    with open(norm_file, 'r') as f:
                        self.normalization_params = json.load(f)
                    logger.info(f"Loaded regression normalization params from {norm_file}")
                except Exception as e:
                    logger.warning(f"Failed to load normalization params from {norm_file}: {e}")
                    self.normalization_params = None
            else:
                logger.debug(f"Normalization params file not found: {norm_file}")

        if not self.tasks: 
            logger.warning(f"Dataset info for split '{self.split}' is missing 'tasks' list or it's empty.")
            if self.samples and 'label' in self.samples[0] and isinstance(self.samples[0]['label'], list):
                
                num_inferred_tasks = len(self.samples[0]['label'])
                self.tasks = [f'task_{i}' for i in range(num_inferred_tasks)]
                logger.info(f"Inferred {num_inferred_tasks} tasks from the first sample. Using tasks: {self.tasks}")
            else:
                
                self.tasks = ['default_task']
                logger.warning(f"Unable to infer tasks from samples. Using default: {self.tasks}")

        logger.info(f"Dataset '{self.split}' initialized with {self.num_samples} samples and {len(self.tasks)} tasks.")
        logger.info(f"Task type: {self.task_type}, Tasks: {self.tasks[:5]}{'...' if len(self.tasks) > 5 else ''}")

        
        
       
        shared_graphs_dir = os.path.join(self.dataset_dir, 'graphs')
        shared_fingerprints_dir = os.path.join(self.dataset_dir, 'fingerprints')
        
        if os.path.exists(shared_graphs_dir):
           
            self.graphs_dir = shared_graphs_dir
            self.fingerprints_dir = shared_fingerprints_dir if os.path.exists(shared_fingerprints_dir) else os.path.join(self.dataset_dir, 'fingerprints', split)
            logger.info(f"Using shared directory structure: graphs_dir={self.graphs_dir}")
        else:
          
            self.graphs_dir = os.path.join(self.dataset_dir, 'graph_data', split) 
            self.fingerprints_dir = os.path.join(self.dataset_dir, 'fingerprints', split)
            logger.info(f"Using standard directory structure: graphs_dir={self.graphs_dir}")
        
        self.conformers_dir = os.path.join(self.dataset_dir, 'conformers')  

      
        missing_dirs = []
        if 'graph' in self.required_features and not os.path.exists(self.graphs_dir):
            missing_dirs.append(f"graphs directory: {self.graphs_dir}")
        if 'fingerprint' in self.required_features and not os.path.exists(self.fingerprints_dir):
            missing_dirs.append(f"fingerprints directory: {self.fingerprints_dir}")
        if 'conformer' in self.required_features and not os.path.exists(self.conformers_dir):
           
            missing_dirs.append(f"conformers directory: {self.conformers_dir}")

        if missing_dirs:
            error_msg = f"Required directories not found for split '{self.split}': {', '.join(missing_dirs)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"All required directories verified for split '{self.split}'.")

        
        from functools import lru_cache
        self._feature_cache = {}  
        self._cache_size = min(1000, self.num_samples // 2)  
        logger.info(f"Initialized feature cache with max size: {self._cache_size}")

    def get_task_info(self) -> Dict[str, Any]:
       
        return {
            'task_type': self.task_type,
            'tasks': self.tasks,
            'num_tasks': len(self.tasks)
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
     
        if idx >= self.num_samples:
            logger.error(f"Index {idx} out of range for dataset size {self.num_samples}")
            return None

        sample_info = self.samples[idx]
        
       
        augmented_sample_id = sample_info.get('augmented_sample_id')
        original_molecule_id_for_files = sample_info.get('original_molecule_id_for_files')
        
        original_molecule_group_id = sample_info.get('original_molecule_group_id', original_molecule_id_for_files)

        if augmented_sample_id is None or original_molecule_id_for_files is None:
            logger.error(f"Sample {idx} missing required IDs. "
                          f"augmented_sample_id: {augmented_sample_id}, "
                        f"original_molecule_id_for_files: {original_molecule_id_for_files}.")
            return None

        sample_data = {}
        feature_load_successful = True
        
     
        cache_key = f"{idx}_{augmented_sample_id}"
        if cache_key in self._feature_cache:
          
            return self._feature_cache[cache_key]

       
        if 'smiles' in self.required_features:
           
            smiles = sample_info.get('smiles', '')
            if smiles:
                sample_data['smiles'] = smiles
            else:
               
                logger.debug(f"Sample {idx} missing SMILES string in metadata.")


       
        if 'graph' in self.required_features:
            try:
              
                graph_file = os.path.join(self.graphs_dir, f"{original_molecule_group_id}.pt")
                
             
                if not os.path.exists(graph_file):
                   
                    graph_file = os.path.join(self.graphs_dir, f"{original_molecule_id_for_files}.pt")
                
                if os.path.exists(graph_file):
                    try:
                        graph_data = torch.load(graph_file, map_location='cpu', weights_only=False)
                        
                     
                        pyg_data = None
                        
                        if isinstance(graph_data, Data):
                          
                            pyg_data = graph_data
                        elif isinstance(graph_data, dict) and 'pyg_data' in graph_data:
                            
                            pyg_data = graph_data['pyg_data']
                        
                        if pyg_data is None:
                            logger.warning(f"Graph file {graph_file} format is unexpected. Expected PyG Data or dict with 'pyg_data'.")
                            feature_load_successful = False
                        elif not isinstance(pyg_data, Data):
                            logger.warning(f"Graph file {graph_file} contains invalid PyG data type: {type(pyg_data)}")
                            feature_load_successful = False
                        else:
                           
                            if not hasattr(pyg_data, 'x') or not hasattr(pyg_data, 'edge_index'):
                                logger.warning(f"Graph file {graph_file} PyG data missing required attributes")
                                feature_load_successful = False
                            else:
                              
                                FIXED_NODE_DIM = 9
                                FIXED_EDGE_DIM = 3
                                
                                
                                if pyg_data.x is not None:
                                    x = pyg_data.x.clone() 
                                    if x.dim() == 1:
                                        x = x.unsqueeze(-1)
                                    if x.size(-1) != FIXED_NODE_DIM:
                                        new_x = torch.zeros(x.size(0), FIXED_NODE_DIM, dtype=x.dtype)
                                        min_dim = min(x.size(-1), FIXED_NODE_DIM)
                                        new_x[:, :min_dim] = x[:, :min_dim]
                                        x = new_x
                                    pyg_data.x = x
                                
                             
                                if pyg_data.edge_index is not None:
                                    pyg_data.edge_index = pyg_data.edge_index.clone()
                                
                                
                                if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
                                    edge_attr = pyg_data.edge_attr.clone() 
                                    if edge_attr.dim() == 1:
                                        edge_attr = edge_attr.unsqueeze(-1)
                                    if edge_attr.size(-1) != FIXED_EDGE_DIM:
                                        new_edge_attr = torch.zeros(edge_attr.size(0), FIXED_EDGE_DIM, dtype=edge_attr.dtype)
                                        min_dim = min(edge_attr.size(-1), FIXED_EDGE_DIM)
                                        new_edge_attr[:, :min_dim] = edge_attr[:, :min_dim]
                                        edge_attr = new_edge_attr
                                    pyg_data.edge_attr = edge_attr
                                elif pyg_data.edge_index is not None:
                                    
                                    num_edges = pyg_data.edge_index.size(1)
                                    pyg_data.edge_attr = torch.zeros(num_edges, FIXED_EDGE_DIM, dtype=torch.float)
                                
                                sample_data['graph_pyg'] = pyg_data
                                
                                logger.debug(f"Sample aug_id={augmented_sample_id}: Using PyG data only (GraphGPS/GNN).")
                    except Exception as e:
                        logger.error(f"Error loading graph file {graph_file}: {e}")
                        feature_load_successful = False
                else:
                    
                    logger.debug(f"Graph file not found (invalid SMILES skipped): {graph_file}")
                    feature_load_successful = False
            except Exception as e:
                logger.error(f"Error loading graph file for sample {idx}: {e}")
                feature_load_successful = False

        
        if 'fingerprint' in self.required_features:
            try:
              
                fingerprint_file = os.path.join(self.fingerprints_dir, f"{original_molecule_group_id}.npz")
                
            
                if not os.path.exists(fingerprint_file):
                    try:
                        org_id_int = int(original_molecule_id_for_files)
                     
                        group_id = org_id_int // 1000
                        item_id = org_id_int % 1000
                        fingerprint_file = os.path.join(self.fingerprints_dir, f"{self.split}_{group_id}_{item_id}.npz")
                    except (ValueError, TypeError):
                
                        pass
                
                if os.path.exists(fingerprint_file):
                    fp_data = np.load(fingerprint_file)
                  
                    if 'fingerprints' in fp_data:
                        fingerprints = torch.from_numpy(fp_data['fingerprints']).float()
                        sample_data['fingerprints'] = fingerprints
                    else:
                        logger.warning(f"Fingerprint file {fingerprint_file} missing 'fingerprints' key.")
                        feature_load_successful = False
                else:
                   
                    logger.debug(f"Fingerprint file not found (invalid SMILES skipped): {fingerprint_file}")
                    feature_load_successful = False
            except Exception as e:
                logger.error(f"Error loading fingerprint file for sample {idx}: {e}")
                feature_load_successful = False

       
        if 'conformer' in self.required_features:
            try:
               
                conformer_file = os.path.join(self.conformers_dir, f"{original_molecule_group_id}.pt")
                
               
                if not os.path.exists(conformer_file):
                 
                    conformer_file = os.path.join(self.conformers_dir, f"{original_molecule_id_for_files}.pt")
                
                if os.path.exists(conformer_file):
                    conformer_data = torch.load(conformer_file, map_location='cpu', weights_only=False)
                    
                 
                    if isinstance(conformer_data, dict):
                      
                        
                        best_conformer = None
                        if 'conformers' in conformer_data and conformer_data['conformers']:
                           
                            best_conformer = conformer_data['conformers'][0]
                        elif 'coordinates' in conformer_data and 'atomic_numbers' in conformer_data:
                           
                            best_conformer = conformer_data
                        
                        if best_conformer is not None:
                            
                       
                            if 'coordinates' in best_conformer:
                                coords = best_conformer['coordinates']
                                if isinstance(coords, np.ndarray):
                                    coords = torch.from_numpy(coords)
                                
                           
                                if not isinstance(coords, torch.Tensor):
                                    logger.warning(f"Conformer file {conformer_file} coordinates is not a tensor: {type(coords)}")
                                    feature_load_successful = False
                                elif coords.dim() != 2 or coords.size(1) != 3:
                                    logger.warning(f"Conformer file {conformer_file} coordinates has invalid shape: {coords.shape}, expected [N, 3]")
                                    feature_load_successful = False
                                elif torch.isnan(coords).any() or torch.isinf(coords).any():
                                    logger.warning(f"Conformer file {conformer_file} coordinates contains NaN or Inf values")
                                    feature_load_successful = False
                                else:
                                    sample_data['conformer_coordinates'] = coords.float()
                            
                         
                            if 'atomic_numbers' in best_conformer:
                                atomic_nums = best_conformer['atomic_numbers']
                                if isinstance(atomic_nums, (list, np.ndarray)):
                                    atomic_nums = torch.tensor(atomic_nums)
                                
                                
                                if not isinstance(atomic_nums, torch.Tensor):
                                    logger.warning(f"Conformer file {conformer_file} atomic_numbers is not a tensor: {type(atomic_nums)}")
                                    feature_load_successful = False
                                elif atomic_nums.dim() != 1:
                                    logger.warning(f"Conformer file {conformer_file} atomic_numbers has invalid shape: {atomic_nums.shape}, expected 1D")
                                    feature_load_successful = False
                                elif atomic_nums.size(0) != sample_data.get('conformer_coordinates', torch.empty(0)).size(0):
                                    logger.warning(f"Conformer file {conformer_file} atomic_numbers count ({atomic_nums.size(0)}) != coordinates count ({sample_data.get('conformer_coordinates', torch.empty(0)).size(0)})")
                                    feature_load_successful = False
                                else:
                                    sample_data['conformer_atomic_numbers'] = atomic_nums.long()
                            
                          
                            if 'num_atoms' in best_conformer:
                                num_atoms = best_conformer['num_atoms']
                                if isinstance(num_atoms, (int, np.integer)):
                                    attention_mask = torch.ones(num_atoms, dtype=torch.bool)
                                    sample_data['conformer_attention_mask'] = attention_mask
                                else:
                                    logger.warning(f"Conformer file {conformer_file} num_atoms is not an integer: {type(num_atoms)}")
                                    feature_load_successful = False
                            elif 'conformer_coordinates' in sample_data:
                               
                                num_atoms = sample_data['conformer_coordinates'].size(0)
                                attention_mask = torch.ones(num_atoms, dtype=torch.bool)
                                sample_data['conformer_attention_mask'] = attention_mask
                            
                           
                            if 'pyg_data' in conformer_data:
                                pyg_data = conformer_data['pyg_data']
                                if isinstance(pyg_data, Data):
                                    
                                    if hasattr(pyg_data, 'pos'):
                                        
                                        if not hasattr(pyg_data, 'x') and hasattr(pyg_data, 'z'):
                                           
                                            num_atoms = pyg_data.pos.size(0)
                                            x = torch.zeros((num_atoms, 9), dtype=torch.float)
                                            x[:, 0] = pyg_data.z.float()  
                                            pyg_data.x = x
                                        elif not hasattr(pyg_data, 'x') and hasattr(pyg_data, 'pos'):
                                            
                                            num_atoms = pyg_data.pos.size(0)
                                            x = torch.zeros((num_atoms, 9), dtype=torch.float)
                                            pyg_data.x = x
                                        
                                      
                                        if hasattr(pyg_data, 'x') and pyg_data.x.size(0) != pyg_data.pos.size(0):
                                            num_atoms = pyg_data.pos.size(0)
                                            x_dim = pyg_data.x.size(1) if pyg_data.x.dim() > 1 else 9
                                            x = torch.zeros((num_atoms, x_dim), dtype=torch.float)
                                            if pyg_data.x.size(0) < num_atoms:
                                                x[:pyg_data.x.size(0)] = pyg_data.x if pyg_data.x.dim() > 1 else pyg_data.x.unsqueeze(1)
                                            else:
                                                x = pyg_data.x[:num_atoms] if pyg_data.x.dim() > 1 else pyg_data.x[:num_atoms].unsqueeze(1)
                                            pyg_data.x = x
                                        
                                       
                                        if hasattr(pyg_data, 'x') and pyg_data.x.dim() == 1:
                                            pyg_data.x = pyg_data.x.unsqueeze(1)
                                            
                                        sample_data['conformer_pyg_data'] = pyg_data
                                    else:
                                        logger.warning(f"Conformer file {conformer_file} PyG data missing pos attribute")
                                else:
                                    logger.warning(f"Conformer file {conformer_file} PyG data is not a Data object: {type(pyg_data)}")
                        else:
                            logger.warning(f"Conformer file {conformer_file} missing valid conformers.")
                            feature_load_successful = False
                    else:
                        logger.warning(f"Conformer file {conformer_file} format is unexpected.")
                        feature_load_successful = False
                else:
                   
                    logger.debug(f"Conformer file not found (invalid SMILES skipped): {conformer_file}")
                    feature_load_successful = False
            except Exception as e:
                logger.error(f"Error loading conformer file for sample {idx}: {e}")
                feature_load_successful = False
        else:
          
            logger.debug(f"Conformer features not required, skipping conformer loading for sample {idx}")

      
        if 'label' in self.required_features:
            try:
                labels = sample_info.get('labels', [])  
                if isinstance(labels, list):
                   
                    labels_tensor = torch.tensor(labels, dtype=torch.float32)
                    sample_data['labels'] = labels_tensor
                else:
                 
                    labels_tensor = torch.tensor([labels], dtype=torch.float32)
                    sample_data['labels'] = labels_tensor
            except Exception as e:
                logger.error(f"Error processing labels for sample {idx}: {e}")
                feature_load_successful = False

       
        sample_data['augmented_sample_id'] = augmented_sample_id

        
        smiles = sample_info.get('smiles', sample_data.get('smiles', ''))
        if smiles:
            try:
                from data.prepare_datasets import smiles_to_descriptors
                descriptors = smiles_to_descriptors(smiles)
                sample_data['descriptors'] = torch.from_numpy(descriptors).float()
            except Exception as e:
                logger.warning(f"Failed to compute descriptors (sample {idx}): {e}")
               
                sample_data['descriptors'] = torch.zeros(30, dtype=torch.float32)
        else:
          
            sample_data['descriptors'] = torch.zeros(30, dtype=torch.float32)

        
        if not feature_load_successful:
            logger.debug(f"Sample aug_id={augmented_sample_id}: Returning None due to feature_load_successful=False.")
            return None

        
        feature_to_required_keys = {
            'smiles': ['smiles'],  
            'graph': ['graph_pyg'],
            'fingerprint': ['fingerprints'],
            'conformer': ['conformer_coordinates'],
            'label': ['labels']
        }
        
        missing_keys = []
        for feature in self.required_features:
            if feature in feature_to_required_keys:
                expected_keys_for_feature = feature_to_required_keys[feature]
                
                
                if feature == 'smiles':
                    if 'smiles' not in sample_data:
                        
                         missing_keys.append('smiles')
                else:
                    
                    for key in expected_keys_for_feature:
                        if key not in sample_data:
                            missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"Sample {idx} missing required keys: {missing_keys}")
            return None

        logger.debug(f"Sample aug_id={augmented_sample_id}: Returning valid sample_data.")
        
        
        if len(self._feature_cache) < self._cache_size:
            self._feature_cache[cache_key] = sample_data
        
        return sample_data


def custom_collate_fn_for_molecules(batch: List[Dict[str, Any]], 
                                   max_nodes: int = 512, 
                                   multi_hop_max_dist: int = 5, 
                                   spatial_pos_max: int = 1024) -> Dict[str, Any]:
    
    
    valid_indices = [i for i, item in enumerate(batch) if item is not None]
    if not valid_indices:
        logger.warning("No valid samples in batch for collation, returning empty batch.")
      
        return {
            'graph_gnn': None,
            'fingerprints': torch.empty((0, 2048), dtype=torch.float),
            'labels': torch.empty((0, 1), dtype=torch.float),
            'augmented_sample_id': [],
            'batch_size': 0
        }
    
    batch = [batch[i] for i in valid_indices]
    logger.debug(f"Batch processing: {len(batch)} valid samples out of original batch")


    graph_pyg_list = []
    fingerprints_list = []
    
    conformer_coordinates_list = []
    conformer_atomic_numbers_list = []
    conformer_attention_mask_list = []
    conformer_pyg_data_list = []
   
    labels_list = []
    augmented_sample_ids_list = []

    
    for data_point in batch:
        essential_keys = ['labels', 'augmented_sample_id']
        missing_essential = [key for key in essential_keys if key not in data_point]
        if missing_essential:
            logger.error(f"Batch sample missing essential keys: {missing_essential}")
            continue

      
        if 'graph_pyg' in data_point:
            graph_pyg_list.append(data_point['graph_pyg'])

      
        if 'fingerprints' in data_point:
            fingerprints_list.append(data_point['fingerprints'])

       
        if 'conformer_coordinates' in data_point:
            conformer_coordinates_list.append(data_point['conformer_coordinates'])
        if 'conformer_atomic_numbers' in data_point:
            conformer_atomic_numbers_list.append(data_point['conformer_atomic_numbers'])
        if 'conformer_attention_mask' in data_point:
            conformer_attention_mask_list.append(data_point['conformer_attention_mask'])
        if 'conformer_pyg_data' in data_point:
            conformer_pyg_data_list.append(data_point['conformer_pyg_data'])

        labels_list.append(data_point['labels'])
        augmented_sample_ids_list.append(data_point['augmented_sample_id'])

    collated_batch = {}

    if graph_pyg_list:
        try:
            valid_graphs = []
            feature_dims = []
            
            for i, graph in enumerate(graph_pyg_list):
                try:
                    if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                        logger.warning(f"Graph {i} missing required attributes (x or edge_index), skipping")
                        continue
                    
                    if not isinstance(graph.x, torch.Tensor) or not isinstance(graph.edge_index, torch.Tensor):
                        logger.warning(f"Graph {i} has invalid tensor types for x or edge_index, skipping")
                        continue
                    
                    if graph.x.size(0) == 0:
                        logger.warning(f"Graph {i} has empty node features (x), skipping")
                        continue
                    
                    if graph.x.dim() != 2:
                        logger.warning(f"Graph {i} has invalid node feature dimensions: {graph.x.shape}, expected 2D tensor, skipping")
                        continue
                    
                    if graph.edge_index.size(1) == 0:
                        logger.debug(f"Graph {i} has no edges, but keeping for batching")
                    
                    if graph.edge_index.size(1) > 0:
                        max_node_idx = graph.edge_index.max().item()
                        if max_node_idx >= graph.x.size(0):
                            logger.warning(f"Graph {i} has edge indices ({max_node_idx}) >= num_nodes ({graph.x.size(0)}), skipping")
                            continue
                    
                    required_attrs = ['edge_attr', 'in_degree', 'out_degree', 'spatial_pos']
                    missing_attrs = []
                    for attr in required_attrs:
                        if not hasattr(graph, attr):
                            missing_attrs.append(attr)
                    
                    if missing_attrs:
                        logger.warning(f"Graph {i} missing attributes: {missing_attrs}, skipping")
                        continue
                    
                    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                        if graph.edge_attr.size(0) != graph.edge_index.size(1):
                            logger.warning(f"Graph {i} edge_attr size ({graph.edge_attr.size(0)}) != edge_index size ({graph.edge_index.size(1)}), skipping")
                            continue
                    
                    valid_graphs.append(graph)
                    feature_dims.append(graph.x.size(1))
                    
                except Exception as graph_error:
                    logger.warning(f"Error validating graph {i}: {graph_error}, skipping")
                    continue
            
            if valid_graphs and feature_dims:
                max_feature_dim = max(feature_dims)
                min_feature_dim = min(feature_dims)
                
                if max_feature_dim != min_feature_dim:
                    logger.info(f"Normalizing PyG graph feature dimensions: {min_feature_dim}-{max_feature_dim} -> {max_feature_dim}")
                
                normalized_graphs = []
                
                for i, graph in enumerate(valid_graphs):
                    try:
                        normalized_graph = Data()
                        
                        current_dim = graph.x.size(1)
                        if current_dim != max_feature_dim:
                            if current_dim < max_feature_dim:
                                padding = torch.zeros(graph.x.size(0), max_feature_dim - current_dim, 
                                                    dtype=graph.x.dtype, device=graph.x.device)
                                normalized_graph.x = torch.cat([graph.x, padding], dim=1)
                            else:
                                normalized_graph.x = graph.x[:, :max_feature_dim].clone()
                        else:
                            normalized_graph.x = graph.x.clone()
                        
                        if graph.edge_index.dtype != torch.long:
                            normalized_graph.edge_index = graph.edge_index.long()
                        else:
                            normalized_graph.edge_index = graph.edge_index.clone()
                        
                        if normalized_graph.edge_index.dim() != 2 or normalized_graph.edge_index.size(0) != 2:
                            num_nodes = normalized_graph.x.size(0)
                            normalized_graph.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(normalized_graph.x.device)
                        
                        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                            if graph.edge_attr.size(0) == graph.edge_index.size(1):
                                normalized_graph.edge_attr = graph.edge_attr.clone()
                            else:
                                edge_dim = graph.edge_attr.size(1) if graph.edge_attr.dim() > 1 else 1
                                normalized_graph.edge_attr = torch.zeros(normalized_graph.edge_index.size(1), edge_dim, 
                                                                      device=normalized_graph.x.device, dtype=graph.edge_attr.dtype)
                        
                        if hasattr(graph, 'in_degree'):
                            normalized_graph.in_degree = graph.in_degree.clone() if isinstance(graph.in_degree, torch.Tensor) else graph.in_degree
                        
                        if hasattr(graph, 'out_degree'):
                            normalized_graph.out_degree = graph.out_degree.clone() if isinstance(graph.out_degree, torch.Tensor) else graph.out_degree
                        
                        
                        for attr_name in ['z', 'pos', 'batch', 'idx']:
                            if hasattr(graph, attr_name):
                                attr_value = getattr(graph, attr_name)
                                if isinstance(attr_value, torch.Tensor):
                                    setattr(normalized_graph, attr_name, attr_value.clone())
                                else:
                                    setattr(normalized_graph, attr_name, attr_value)
                        
                        normalized_graphs.append(normalized_graph)
                    
                    except Exception as norm_error:
                        logger.warning(f"Error normalizing graph {i}: {norm_error}, skipping")
                        continue
                
                valid_graphs = normalized_graphs
            
            if valid_graphs:
                try:
                    for i, graph in enumerate(valid_graphs):
                        if graph.edge_index.dtype != torch.long:
                            graph.edge_index = graph.edge_index.long()
                        
                        if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
                            logger.warning(f"Graph {i} has invalid edge_index shape: {graph.edge_index.shape}, fixing...")
                            num_nodes = graph.x.size(0)
                            graph.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(graph.x.device)
                    
                    collated_batch['graph_gnn'] = PyGBatch.from_data_list(valid_graphs)
                except Exception as batch_error:
                    logger.warning(f"PyG batching failed: {batch_error}. Skipping PyG graphs for this batch.")
                    try:
                        if len(valid_graphs) > 0:
                            template = valid_graphs[0]
                            batch_size = len(valid_graphs)
                            max_nodes = max(g.x.size(0) for g in valid_graphs)
                            feature_dim = template.x.size(1)
                            
                            dummy_batch = Data()
                            dummy_batch.x = torch.zeros(batch_size * max_nodes, feature_dim, device=template.x.device)
                            dummy_batch.edge_index = torch.zeros(2, batch_size * max_nodes, device=template.x.device, dtype=torch.long)
                            dummy_batch.batch = torch.repeat_interleave(torch.arange(batch_size, device=template.x.device), max_nodes)
                            
                            node_offset = 0
                            for i, graph in enumerate(valid_graphs):
                                num_nodes = graph.x.size(0)
                                dummy_batch.x[node_offset:node_offset+num_nodes] = graph.x
                                
                                dummy_batch.edge_index[0, node_offset:node_offset+num_nodes] = torch.arange(node_offset, node_offset+num_nodes, device=template.x.device)
                                dummy_batch.edge_index[1, node_offset:node_offset+num_nodes] = torch.arange(node_offset, node_offset+num_nodes, device=template.x.device)
                                
                                node_offset += num_nodes
                            
                            collated_batch['graph_gnn'] = dummy_batch
                            logger.info(f"Created fallback dummy batch with {batch_size} graphs")
                        else:
                            collated_batch['graph_gnn'] = None
                    except Exception as e:
                        logger.error(f"Failed to create fallback batch: {e}")
                        collated_batch['graph_gnn'] = None
            else:
                logger.warning("No valid graphs for PyG batching")
                collated_batch['graph_gnn'] = None
        except Exception as e:
            logger.error(f"Error batching PyG graphs: {e}")
            collated_batch['graph_gnn'] = None

    
    if fingerprints_list:
        try:
            collated_batch['fingerprints'] = torch.stack(fingerprints_list)
            logger.debug(f"Fingerprints batched successfully. Shape: {collated_batch['fingerprints'].shape}")
        except Exception as e:
            logger.error(f"Error batching fingerprints: {e}")

    if conformer_coordinates_list and conformer_atomic_numbers_list:
        try:
            max_atoms = max(coords.shape[0] for coords in conformer_coordinates_list)
            batch_size = len(conformer_coordinates_list)
            
            padded_coordinates = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
            padded_atomic_numbers = torch.zeros(batch_size, max_atoms, dtype=torch.long)
            padded_attention_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
            
            for i, (coords, atomic_nums) in enumerate(zip(conformer_coordinates_list, conformer_atomic_numbers_list)):
                num_atoms = coords.shape[0]
                padded_coordinates[i, :num_atoms, :] = coords
                padded_atomic_numbers[i, :num_atoms] = atomic_nums
                padded_attention_mask[i, :num_atoms] = True
            
            collated_batch['conformer_coordinates'] = padded_coordinates
            collated_batch['conformer_atomic_numbers'] = padded_atomic_numbers
            collated_batch['conformer_attention_mask'] = padded_attention_mask
            
            logger.debug(f"3D conformers batched successfully. Shape: {padded_coordinates.shape}")
        except Exception as e:
            logger.error(f"Error batching 3D conformers: {e}")

    if conformer_pyg_data_list:
        try:
            valid_conformers = []
            feature_dims = []
            
            for i, conf in enumerate(conformer_pyg_data_list):
                try:
                    if not hasattr(conf, 'pos') or not hasattr(conf, 'x'):
                        logger.warning(f"Conformer {i} missing required attributes (pos or x), skipping")
                        continue
                    
                    if not isinstance(conf.pos, torch.Tensor) or not isinstance(conf.x, torch.Tensor):
                        logger.warning(f"Conformer {i} has invalid tensor types for pos or x, skipping")
                        continue
                    
                    if conf.x.size(0) == 0:
                        logger.warning(f"Conformer {i} has empty node features (x), skipping")
                        continue
                    
                    if conf.pos.size(0) == 0:
                        logger.warning(f"Conformer {i} has empty positions (pos), skipping")
                        continue
                    
                    if conf.x.dim() != 2:
                        logger.warning(f"Conformer {i} has invalid node feature dimensions: {conf.x.shape}, expected 2D tensor, skipping")
                        continue
                    
                    if conf.pos.dim() != 2 or conf.pos.size(1) != 3:
                        logger.warning(f"Conformer {i} has invalid position dimensions: {conf.pos.shape}, expected [N, 3], skipping")
                        continue
                    
                    if conf.x.size(0) != conf.pos.size(0):
                        logger.warning(f"Conformer {i} has inconsistent node counts: x({conf.x.size(0)}) != pos({conf.pos.size(0)}), skipping")
                        continue
                    
                    if conf.x.dtype not in [torch.float32, torch.float64]:
                        logger.warning(f"Conformer {i} has invalid x dtype: {conf.x.dtype}, expected float32/float64, skipping")
                        continue
                    
                    if conf.pos.dtype not in [torch.float32, torch.float64]:
                        logger.warning(f"Conformer {i} has invalid pos dtype: {conf.pos.dtype}, expected float32/float64, skipping")
                        continue
                    
                    if torch.isnan(conf.x).any() or torch.isinf(conf.x).any():
                        logger.warning(f"Conformer {i} has NaN or Inf values in x, skipping")
                        continue
                    
                    if torch.isnan(conf.pos).any() or torch.isinf(conf.pos).any():
                        logger.warning(f"Conformer {i} has NaN or Inf values in pos, skipping")
                        continue
                    
                    valid_conformers.append(conf)
                    feature_dims.append(conf.x.size(1))
                    
                except Exception as conf_error:
                    logger.warning(f"Error validating conformer {i}: {conf_error}, skipping")
                    continue
            
            if valid_conformers and feature_dims:
                max_feature_dim = max(feature_dims)
                min_feature_dim = min(feature_dims)
                
                if max_feature_dim != min_feature_dim:
                    logger.info(f"Normalizing conformer PyG feature dimensions: {min_feature_dim}-{max_feature_dim} -> {max_feature_dim}")
                
                normalized_conformers = []
                
                for i, conf in enumerate(valid_conformers):
                    try:
                        normalized_conf = Data()
                        
                        current_dim = conf.x.size(1)
                        if current_dim != max_feature_dim:
                            if current_dim < max_feature_dim:
                                padding = torch.zeros(conf.x.size(0), max_feature_dim - current_dim, 
                                                    dtype=conf.x.dtype, device=conf.x.device)
                                normalized_conf.x = torch.cat([conf.x, padding], dim=1)
                            else:
                                normalized_conf.x = conf.x[:, :max_feature_dim].clone()
                        else:
                            normalized_conf.x = conf.x.clone()
                        
                        if hasattr(conf, 'pos') and conf.pos is not None:
                            if conf.pos.size(0) == normalized_conf.x.size(0):
                                normalized_conf.pos = conf.pos.clone()
                            else:
                                logger.warning(f"Conformer {i} has inconsistent node counts between pos and x, adjusting...")
                                min_nodes = min(conf.pos.size(0), normalized_conf.x.size(0))
                                if conf.pos.size(0) > min_nodes:
                                    normalized_conf.pos = conf.pos[:min_nodes].clone()
                                    normalized_conf.x = normalized_conf.x[:min_nodes]
                                else:
                                    normalized_conf.x = normalized_conf.x[:min_nodes]
                                    normalized_conf.pos = conf.pos.clone()
                        else:
                            logger.warning(f"Conformer {i} is missing the pos attribute, creating random positions...")
                            num_nodes = normalized_conf.x.size(0)
                            normalized_conf.pos = torch.randn(num_nodes, 3, device=normalized_conf.x.device)
                        
                        if hasattr(conf, 'edge_index') and conf.edge_index is not None:
                            if conf.edge_index.dtype != torch.long:
                                normalized_conf.edge_index = conf.edge_index.long()
                            else:
                                normalized_conf.edge_index = conf.edge_index.clone()
                            
                           
                            if normalized_conf.edge_index.dim() != 2 or normalized_conf.edge_index.size(0) != 2:
                                num_nodes = normalized_conf.x.size(0)
                                normalized_conf.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(normalized_conf.x.device)
                               
                            
                           
                            if normalized_conf.edge_index.numel() > 0:
                                max_idx = normalized_conf.edge_index.max().item()
                                if max_idx >= normalized_conf.x.size(0):
                                 
                                    num_nodes = normalized_conf.x.size(0)
                                    normalized_conf.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(normalized_conf.x.device)
                                  
                        else:
                           
                            num_nodes = normalized_conf.x.size(0)
                            normalized_conf.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(normalized_conf.x.device)
                        
                        
                        for attr_name in ['z', 'edge_attr', 'batch', 'idx']:
                            if hasattr(conf, attr_name):
                                attr_value = getattr(conf, attr_name)
                                if isinstance(attr_value, torch.Tensor):
                                    setattr(normalized_conf, attr_name, attr_value.clone())
                                else:
                                    setattr(normalized_conf, attr_name, attr_value)
                        
                       
                        if not hasattr(normalized_conf, 'edge_attr') or normalized_conf.edge_attr is None:
                           
                            num_edges = normalized_conf.edge_index.size(1)
                            normalized_conf.edge_attr = torch.zeros(num_edges, 3, dtype=torch.float, device=normalized_conf.x.device)
                        elif normalized_conf.edge_attr.size(0) != normalized_conf.edge_index.size(1):
                           
                            num_edges = normalized_conf.edge_index.size(1)
                            normalized_conf.edge_attr = torch.zeros(num_edges, 3, dtype=torch.float, device=normalized_conf.x.device)
                        
                    
                        normalized_conformers.append(normalized_conf)
                    
                    except Exception as norm_error:
                        logger.warning(f"Error normalizing conformer {i}: {norm_error}, skipping")
                        continue
                
                valid_conformers = normalized_conformers
            
            if valid_conformers:
                try:
                    for i, conf in enumerate(valid_conformers):
                        if not hasattr(conf, 'edge_index') or conf.edge_index is None:
                            num_nodes = conf.pos.size(0)
                            conf.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(conf.pos.device)
                        else:
                            if conf.edge_index.dtype != torch.long:
                                conf.edge_index = conf.edge_index.long()
                            
                           
                            if conf.edge_index.dim() != 2 or conf.edge_index.size(0) != 2:
                                logger.warning(f"Conformer {i} has invalid edge_index shape: {conf.edge_index.shape}, fixing...")
                             
                                num_nodes = conf.pos.size(0)
                                conf.edge_index = torch.arange(num_nodes).repeat(2, 1).long().to(conf.pos.device)
                    
                    collated_batch['conformer_pyg_data'] = PyGBatch.from_data_list(valid_conformers)
                except Exception as batch_error:
                    logger.warning(f"Conformer PyG batching failed: {batch_error}. Skipping conformer PyG data for this batch.")
                    try:
                        if len(valid_conformers) > 0:
                            template = valid_conformers[0]
                            batch_size = len(valid_conformers)
                            max_nodes = max(c.pos.size(0) for c in valid_conformers)
                            feature_dim = template.x.size(1)
                            
                            dummy_batch = Data()
                            dummy_batch.x = torch.zeros(batch_size * max_nodes, feature_dim, device=template.pos.device)
                            dummy_batch.pos = torch.zeros(batch_size * max_nodes, 3, device=template.pos.device)
                            dummy_batch.edge_index = torch.zeros(2, batch_size * max_nodes, device=template.pos.device, dtype=torch.long)
                            dummy_batch.batch = torch.repeat_interleave(torch.arange(batch_size, device=template.pos.device), max_nodes)
                            
                            
                            node_offset = 0
                            for i, conf in enumerate(valid_conformers):
                                num_nodes = conf.pos.size(0)
                                dummy_batch.x[node_offset:node_offset+num_nodes] = conf.x
                                dummy_batch.pos[node_offset:node_offset+num_nodes] = conf.pos
                                
                               
                                dummy_batch.edge_index[0, node_offset:node_offset+num_nodes] = torch.arange(node_offset, node_offset+num_nodes, device=template.pos.device)
                                dummy_batch.edge_index[1, node_offset:node_offset+num_nodes] = torch.arange(node_offset, node_offset+num_nodes, device=template.pos.device)
                                
                                node_offset += num_nodes
                            
                            collated_batch['conformer_pyg_data'] = dummy_batch
                            logger.info(f"Created fallback dummy conformer batch with {batch_size} conformers")
                        else:
                            collated_batch['conformer_pyg_data'] = None
                    except Exception as e:
                        logger.error(f"Failed to create fallback conformer batch: {e}")
                        collated_batch['conformer_pyg_data'] = None
            else:
                logger.warning("No valid conformers for PyG batching")
                collated_batch['conformer_pyg_data'] = None
        except Exception as e:
            logger.error(f"Error batching conformer PyG data: {e}")
            collated_batch['conformer_pyg_data'] = None

    
    if labels_list:
        try:
            collated_batch['labels'] = torch.stack(labels_list)
            logger.debug(f"Labels batched successfully. Shape: {collated_batch['labels'].shape}")
        except Exception as e:
            logger.error(f"Error batching labels: {e}")

    if augmented_sample_ids_list:
        try:
            collated_batch['augmented_sample_ids'] = augmented_sample_ids_list
            logger.debug(f"Successfully processed Augmented Sample ID batch of size {len(augmented_sample_ids_list)}.")
        except Exception as e_aug_ids:
            logger.error(f"Error processing augmented_sample_ids: {e_aug_ids}.", exc_info=True)

    if not collated_batch:
        logger.error("Collated batch is empty. No features were successfully batched.")
        return {}

    return collated_batch

def get_dataset_info(dataset_name: str, data_dir: str) -> Dict[str, Any]:
    processed_dataset_dir = os.path.join(data_dir, 'processed', dataset_name)
    info_file = os.path.join(processed_dataset_dir, 'info.json')
    
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Dataset info file not found: {info_file}")
    
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)
    
    return dataset_info


_RENDER_CACHE = {}
_RENDERER = None

def ghmf_collate_fn(batch: List[Dict[str, Any]], dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    
    global _RENDER_CACHE, _RENDERER

    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        logger.warning("GHMF collate: No valid samples in batch")
        return {}

    
    complete_samples = []
    for item in valid_batch:
        has_graph = 'graph_pyg' in item and item['graph_pyg'] is not None
        has_fp = 'fingerprints' in item and item['fingerprints'] is not None
        has_labels = 'labels' in item and item['labels'] is not None
        has_conf = ('conformer_coordinates' in item and
                    item['conformer_coordinates'] is not None and
                    'conformer_atomic_numbers' in item and
                    item['conformer_atomic_numbers'] is not None)

        if has_graph and has_fp and has_labels and has_conf:
            complete_samples.append(item)
        else:
            logger.debug(f"GHMF collate: Skipping incomplete sample (graph={has_graph}, fp={has_fp}, labels={has_labels}, conformer={has_conf})")
    
    if not complete_samples:
        logger.warning("GHMF collate: No complete samples in batch after filtering")
        return {}
    
    valid_batch = complete_samples

    collated = {}

    graph_pyg_list = []
    labels_list = []
    conformer_coords_list = []
    conformer_z_list = []
    descriptors_list = []
    fingerprints_list = []  
    sample_ids = []
    smiles_list = [] 
    has_conformer_list = []
    sample_idx_to_graph_idx = {}
    current_graph_idx = 0

    for i, item in enumerate(valid_batch):
        graph_pyg_list.append(item['graph_pyg'])
        sample_idx_to_graph_idx[i] = current_graph_idx
        current_graph_idx += 1

        if 'smiles' in item and item['smiles'] is not None:
            smiles_list.append(item['smiles'])

        labels_list.append(item['labels'])

        if 'descriptors' in item and item['descriptors'] is not None:
            descriptors_list.append(item['descriptors'])

        fingerprints_list.append(item['fingerprints'])

        
        has_conformer = ('conformer_coordinates' in item and
                        item['conformer_coordinates'] is not None and
                        'conformer_atomic_numbers' in item and
                        item['conformer_atomic_numbers'] is not None)
        has_conformer_list.append(has_conformer)

        if has_conformer:
            conformer_coords_list.append(item['conformer_coordinates'])
            conformer_z_list.append(item['conformer_atomic_numbers'])

        
        if 'augmented_sample_id' in item:
            sample_ids.append(item['augmented_sample_id'])
    
    
    if graph_pyg_list and PYG_AVAILABLE:
        try:
            from torch_geometric.data import Batch as PyGBatch, Data
            
           
            valid_graphs = []
            FIXED_NODE_DIM = 9  
            FIXED_EDGE_DIM = 3  
            
            for i, graph in enumerate(graph_pyg_list):
                if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                    if graph.x is not None and graph.edge_index is not None:
                       
                        new_graph = Data()
                        
                        
                        x = graph.x.clone() if graph.x.is_leaf else graph.x.detach().clone()
                        if x.dim() == 1:
                            x = x.unsqueeze(-1)
                        if x.size(-1) != FIXED_NODE_DIM:
                            new_x = torch.zeros(x.size(0), FIXED_NODE_DIM, dtype=x.dtype, device=x.device)
                            min_dim = min(x.size(-1), FIXED_NODE_DIM)
                            new_x[:, :min_dim] = x[:, :min_dim]
                            x = new_x
                        new_graph.x = x
                        
                       
                        new_graph.edge_index = graph.edge_index.clone()
                        
                        
                        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                            edge_attr = graph.edge_attr.clone() if graph.edge_attr.is_leaf else graph.edge_attr.detach().clone()
                            if edge_attr.dim() == 1:
                                edge_attr = edge_attr.unsqueeze(-1)
                            if edge_attr.size(-1) != FIXED_EDGE_DIM:
                                new_edge_attr = torch.zeros(edge_attr.size(0), FIXED_EDGE_DIM, dtype=edge_attr.dtype, device=edge_attr.device)
                                min_dim = min(edge_attr.size(-1), FIXED_EDGE_DIM)
                                new_edge_attr[:, :min_dim] = edge_attr[:, :min_dim]
                                edge_attr = new_edge_attr
                            new_graph.edge_attr = edge_attr
                        else:
                           
                            num_edges = new_graph.edge_index.size(1)
                            new_graph.edge_attr = torch.zeros(num_edges, FIXED_EDGE_DIM, dtype=torch.float)
                        
                        valid_graphs.append(new_graph)
            
            if valid_graphs:
                batched_graph = PyGBatch.from_data_list(valid_graphs)
                
             
                collated['node_features'] = batched_graph.x
                collated['edge_index'] = batched_graph.edge_index
                collated['batch'] = batched_graph.batch
                collated['edge_attr'] = batched_graph.edge_attr
                
             
                collated['graph_gnn'] = batched_graph
                
        except Exception as e:
            logger.error(f"GHMF collate: Error batching graphs: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
   
    if conformer_coords_list:
        try:
            all_pos = []
            all_z = []
            pos_batch = []

           
            conformer_idx = 0

           
            for i, has_conformer in enumerate(has_conformer_list):
                if has_conformer:
                    coords = conformer_coords_list[conformer_idx]
                    z = conformer_z_list[conformer_idx]
                    conformer_idx += 1

                    
                    if coords.size(0) != z.size(0):
                        logger.warning(f"Sample {i}: coords.size={coords.size(0)} != z.size={z.size(0)}, skipping")
                        continue

                    
                    if i in sample_idx_to_graph_idx:
                        all_pos.append(coords)
                        all_z.append(z)

                       
                        graph_idx = sample_idx_to_graph_idx[i]
                        pos_batch.extend([graph_idx] * coords.size(0))

            if all_pos:
                collated['pos'] = torch.cat(all_pos, dim=0)
                collated['pos_batch'] = torch.tensor(pos_batch, dtype=torch.long)

            if all_z:
                collated['z'] = torch.cat(all_z, dim=0)
               
                collated['z'] = collated['z'].clamp(0, 99)
            
            
            try:
                
                if _RENDERER is None:
                    from models.multiview_renderer import create_multiview_renderer
                    
                    _RENDERER = create_multiview_renderer({
                        'image_size': 64,
                        'background_color': [255, 255, 255],
                        'atom_scale': 0.3,
                        'bond_width': 2,
                        'padding': 0.15
                    })
                    logger.debug("🎨 Initialized global MultiViewRenderer with image_size=64")

               
                batch_rendered_images = []
                
               
                
                for item in valid_batch:
                   
                    has_conf = ('conformer_coordinates' in item and 
                               item['conformer_coordinates'] is not None and 
                               'conformer_atomic_numbers' in item and
                               item['conformer_atomic_numbers'] is not None)
                    
                    if has_conf:
                        aug_id = item.get('augmented_sample_id')
                        
                       
                        if aug_id in _RENDER_CACHE:
                            batch_rendered_images.append(_RENDER_CACHE[aug_id])
                        else:
                           
                            disk_cache_hit = False
                            if aug_id is not None:
                                try:
                                   
                                    if dataset_dir:
                                        cache_dir = os.path.join(dataset_dir, 'cache', 'rendered_images')
                                    else:
                                        cache_dir = os.path.join(PROCESSED_DIR, 'cache', 'rendered_images')
                                    safe_id = str(aug_id).replace('/', '_').replace('\\', '_')
                                    cache_path = os.path.join(cache_dir, f"{safe_id}.pt")
                                    
                                    if os.path.exists(cache_path):
                                       
                                        img = torch.load(cache_path, map_location='cpu', weights_only=False)
                                        _RENDER_CACHE[aug_id] = img
                                        batch_rendered_images.append(img)
                                        disk_cache_hit = True
                                except Exception:
                                    pass
                            
                            if not disk_cache_hit:
                                
                                coords = item['conformer_coordinates']
                                z = item['conformer_atomic_numbers']
                                
                                
                                if coords.size(0) != z.size(0):
                                    logger.warning(f"Data mismatch: coords atoms={coords.size(0)} != z atoms={z.size(0)}, aug_id={aug_id}. Skipping rendering.")
                                   
                                    img = torch.zeros(6, 3, 64, 64, dtype=torch.float32)
                                    batch_rendered_images.append(img)
                                    continue
                                
                               
                                num_atoms = coords.size(0)
                                if num_atoms > 100:
                                    
                                    img = torch.zeros(6, 3, 64, 64, dtype=torch.float32)
                                else:
                                   
                                    img = _RENDERER.render_molecule_torch(coords, z)
                                
                                
                                if aug_id is not None:
                                    _RENDER_CACHE[aug_id] = img
                                    
                                   
                                    try:
                                        if dataset_dir:
                                            cache_dir = os.path.join(dataset_dir, 'cache', 'rendered_images')
                                        else:
                                            cache_dir = os.path.join(PROCESSED_DIR, 'cache', 'rendered_images')
                                        os.makedirs(cache_dir, exist_ok=True)
                                        
                                        safe_id = str(aug_id).replace('/', '_').replace('\\', '_')
                                        cache_path = os.path.join(cache_dir, f"{safe_id}.pt")
                                        
                                        torch.save(img.clone(), cache_path)
                                    except Exception as cache_err:
                                        pass
                                
                                batch_rendered_images.append(img)
                
                if batch_rendered_images:
                    collated['rendered_images'] = torch.stack(batch_rendered_images, dim=0)
                
            except Exception as render_e:
                logger.error(f"GHMF collate: Rendering failed: {render_e}")

        except Exception as e:
            logger.error(f"GHMF collate: Error batching 3D coordinates: {e}")
    
    if labels_list:
        try:
            collated['labels'] = torch.stack(labels_list)
        except Exception as e:
            logger.error(f"GHMF collate: Error batching labels: {e}")
    
    
    if descriptors_list:
        try:
            collated['descriptors'] = torch.stack(descriptors_list)
        except Exception as e:
            logger.error(f"GHMF collate: Error batching descriptors: {e}")
    
    if fingerprints_list:
        try:
            collated['fingerprints'] = torch.stack(fingerprints_list)
        except Exception as e:
            logger.error(f"GHMF collate: Error batching fingerprints: {e}")
    
    if sample_ids:
        collated['sample_ids'] = sample_ids
    
    if smiles_list:
        collated['smiles'] = smiles_list
    
    return collated


def seed_dataloader_worker(worker_id: int):
    """Module-level callback so DataLoader can pickle it under spawn."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_ghmf_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    required_features: List[str] = ['smiles', 'graph', 'fingerprint', 'conformer', 'label'],
    seed: int = 42,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
   

    from data.prepare_datasets import load_and_prepare_dataset
    from functools import partial
    
    try:
        prepared_data, dataset_info = load_and_prepare_dataset(dataset_name, data_dir)
        
        processed_dataset_dir = os.path.join(data_dir, 'processed', dataset_name)
        loaders = {}
        base_generator = torch.Generator()
        base_generator.manual_seed(int(seed))
        
        for split in ['train', 'val', 'test']:
            if prepared_data.get(split):
                dataset = MoleculeDataset(
                    samples=prepared_data[split],
                    dataset_dir=processed_dataset_dir,
                    task_info=dataset_info,
                    split=split,
                    required_features=required_features,
                )
                
                shuffle = (split == 'train')
                
                collate_fn_with_dataset = partial(ghmf_collate_fn, dataset_dir=processed_dataset_dir)
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=collate_fn_with_dataset,
                    pin_memory=torch.cuda.is_available(),
                    worker_init_fn=seed_dataloader_worker,
                    generator=base_generator,
                )
                
                loaders[split] = loader
                logger.info(f"✅ GHMF {split} loader: {len(loader)} batches")
            else:
                loaders[split] = None
        
        return loaders['train'], loaders['val'], loaders['test']
        
    except Exception as e:
        logger.error(f"Error creating GHMF dataloaders: {e}")
        return None, None, None

create_dataloaders = create_ghmf_dataloaders
