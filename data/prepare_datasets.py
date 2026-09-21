import sys
import os
from torch_geometric.data import Data
import torch
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

import pandas as pd
import numpy as np
import argparse
import logging
import shutil
import inspect
import hashlib
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import json
import yaml

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

import deepchem as dc

logging.getLogger('deepchem.feat.base_classes').setLevel(logging.ERROR)
logging.getLogger('deepchem.data.data_loader').setLevel(logging.ERROR)
logging.getLogger('deepchem.molnet.load_function.molnet_loader').setLevel(logging.ERROR)
logging.getLogger('deepchem.utils.data_utils').setLevel(logging.ERROR)
from deepchem.utils.data_utils import get_data_dir

from torch_geometric.utils import degree

import networkx as nx


from data.split_utils import scaffold_split as _deterministic_scaffold_split

from data.conformer_generator import (
    CONFORMER_MAX_ITERS,
    generate_single_conformer,
    print_conformer_generation_stats,
)


def scaffold_split(
    smiles_list: List[str],
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    include_chirality: bool = True,
    print_metrics: bool = False,
    **kwargs,
) -> Tuple[List[int], List[int], List[int]]:
    del print_metrics, kwargs
    if not include_chirality:
        logger.warning("The current scaffold_split implementation always includes chirality information.")
    return _deterministic_scaffold_split(
        smiles_list,
        frac_train=train_frac,
        frac_valid=valid_frac,
        frac_test=test_frac,
    )


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


logging.getLogger("data.split_utils").setLevel(logging.INFO)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def clean_invalid_smiles(df: pd.DataFrame, smiles_col: str = 'smiles', 
                          output_dir: Optional[str] = None,
                          dataset_name: str = 'unknown') -> Tuple[pd.DataFrame, List[Dict]]:
    
    if smiles_col not in df.columns:
        logger.warning(f"clean_invalid_smiles: column '{smiles_col}' is not in the DataFrame; skipping cleaning.")
        return df, []
    
    original_count = len(df)
    invalid_samples = []
    valid_mask = []
    
   
    
    logger.info(f"Starting SMILES cleaning... original samples: {original_count}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating SMILES"):
        smiles = row[smiles_col]
        is_valid = True
        reason = ""
        
        if not isinstance(smiles, str) or not smiles.strip():
            is_valid = False
            reason = "empty_or_invalid_type"
        else:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    is_valid = False
                    reason = "rdkit_parse_failed"
                else:
                    try:
                        mol_with_h = Chem.AddHs(mol)
                        if mol_with_h is None:
                            is_valid = False
                            reason = "addhs_failed"
                        else:
                            
                            try:
                                AllChem.Compute2DCoords(mol_with_h)
                            except Exception as e_2d:
                             
                                is_valid = False
                                reason = f"compute2d_failed: {str(e_2d)[:50]}"
                    except Exception as e_addhs:
                        is_valid = False
                        reason = f"addhs_exception: {str(e_addhs)[:50]}"
            except Exception as e_parse:
                is_valid = False
                reason = f"parse_exception: {str(e_parse)[:50]}"
        
        valid_mask.append(is_valid)
        
        if not is_valid:
            invalid_samples.append({
                'original_index': idx,
                'smiles': smiles if isinstance(smiles, str) else str(smiles),
                'reason': reason
            })
    
   
    df_cleaned = df[valid_mask].copy()
    
   
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    cleaned_count = len(df_cleaned)
    removed_count = original_count - cleaned_count
    removal_rate = (removed_count / original_count * 100.0) if original_count else 0.0
    
    logger.info(f"SMILES cleaning completed: removed {removed_count} invalid samples ({removal_rate:.2f}%)")
    logger.info(f"Samples after cleaning: {cleaned_count}")
    
    if output_dir and invalid_samples:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f'cleaned_samples_{dataset_name}.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_name,
                'original_count': original_count,
                'cleaned_count': cleaned_count,
                'removed_count': removed_count,
                'removed_samples': invalid_samples
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Cleaning log saved to: {log_file}")
    
    if invalid_samples:
        logger.info(f"Examples of cleaned samples (first 10):")
        for sample in invalid_samples[:10]:
            logger.info(f"  - idx={sample['original_index']}, reason={sample['reason']}, smiles={sample['smiles'][:60]}...")
    
    return df_cleaned, invalid_samples

def filter_outliers(df: pd.DataFrame, 
                   task_type: str, 
                   tasks_list: List[str], 
                   max_atoms: int = 150, 
                   remove_label_outliers: bool = False) -> Tuple[pd.DataFrame, Dict]:
  
    initial_count = len(df)
    logger.info(f"--- Starting statistical outlier filtering ---")
    logger.info(f"Original count: {initial_count}, Max atoms: {max_atoms}, remove label outliers: {remove_label_outliers}")
    
    keep_mask = np.ones(len(df), dtype=bool)
    
    
    logger.info("Checking molecular sizes...")
    for idx, smiles in enumerate(df['smiles']):
        if not keep_mask[idx]: continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                num_heavy = mol.GetNumHeavyAtoms()
                if num_heavy > max_atoms:
                    keep_mask[idx] = False
                elif num_heavy == 0:
                    keep_mask[idx] = False
            else:
                keep_mask[idx] = False 
        except:
            keep_mask[idx] = False
    
    removed_by_size = initial_count - np.sum(keep_mask)
    if removed_by_size > 0:
        logger.warning(f"⚠️ Molecular-size filter (> {max_atoms} atoms): removed {removed_by_size} samples")
    else:
        logger.info("✅ No molecules were removed due to size.")

    removed_by_label = 0
    if task_type == 'regression' and remove_label_outliers and len(df) > 50:
        current_mask = keep_mask.copy()
        for task in tasks_list:
            if task not in df.columns: continue
            
        
            values = pd.to_numeric(df[task], errors='coerce')
            valid_vals = values[current_mask & values.notna()]
            
            if len(valid_vals) > 10:
                mean = valid_vals.mean()
                std = valid_vals.std()
                if std < 1e-6: continue 
                
                
                lower = mean - 4 * std 
                upper = mean + 4 * std
                
                
                outliers = (values < lower) | (values > upper)
                outliers = outliers & values.notna() & current_mask
                
                count = outliers.sum()
                if count > 0:
                    keep_mask = keep_mask & (~outliers)
                    removed_by_label += count
                    logger.warning(f"⚠️ Task {task}: removed {count} label outliers (range [{lower:.2f}, {upper:.2f}])")

    df_filtered = df[keep_mask].copy().reset_index(drop=True)
    final_count = len(df_filtered)
    
    stats = {
        'original': initial_count,
        'final': final_count,
        'removed_total': initial_count - final_count,
        'removed_by_size': removed_by_size,
        'removed_by_label': removed_by_label
    }
    
    removal_rate = ((initial_count - final_count) / initial_count * 100.0) if initial_count else 0.0
    logger.info(f"Outlier filtering completed. Remaining: {final_count} (removal rate: {removal_rate:.2f}%)")
    logger.info(f"--------------------------------------------------")
    return df_filtered, stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def get_available_loaders():
  
    loaders = {}
    for name, obj in inspect.getmembers(dc.molnet):
        if name.startswith('load_'):
            dataset_name = name[5:].lower()  
            loaders[dataset_name] = obj
    return loaders

AVAILABLE_LOADERS = get_available_loaders()

CLASSIFICATION_DATASETS = ['bace', 'bbbp', 'tox21', 'sider', 'clintox']
REGRESSION_DATASETS = ['esol', 'freesolv', 'lipo', 'lipophilicity']
SUPPORTED_DATASETS = set(CLASSIFICATION_DATASETS + REGRESSION_DATASETS)

def get_loader_function_for_dataset(dataset_name):
    
    dataset_mapping = {
        'bace': ['load_bace_classification', 'load_bace'],
        'bbbp': ['load_bbbp'],
        'tox21': ['load_tox21'],
        'sider': ['load_sider'],
        'clintox': ['load_clintox'],
        'esol': ['load_delaney', 'load_esol'],
        'freesolv': ['load_sampl', 'load_freesolv'],
        'lipo': ['load_lipophilicity', 'load_lipo'],
        'lipophilicity': ['load_lipophilicity', 'load_lipo'],
    }
    
    dataset_key = dataset_name.lower()
    if dataset_key not in SUPPORTED_DATASETS:
        logger.error(f"Unsupported dataset: {dataset_name}. Supported datasets: {sorted(SUPPORTED_DATASETS)}")
        return None

    if dataset_key in dataset_mapping:
        for func_name in dataset_mapping[dataset_key]:
            func_variants = [func_name]
            if not func_name.startswith("load_"):
                func_variants.append(f"load_{func_name}")
            else:
                func_variants.append(func_name[5:]) 

            for variant in func_variants:
                if variant in AVAILABLE_LOADERS:
                    logger.info(f"Found loader {variant} for dataset {dataset_name}")
                    return AVAILABLE_LOADERS[variant]

    
    general_func_name = f"load_{dataset_key}"
    if general_func_name in AVAILABLE_LOADERS:
        logger.info(f"Using {general_func_name} to load dataset {dataset_name}")
        return AVAILABLE_LOADERS[general_func_name]
    
    logger.error(f"No loader found for dataset {dataset_name}")
    return None

def smiles_to_descriptors(smiles: str) -> np.ndarray:
   
    default_descriptors = np.zeros(30, dtype=np.float32)
    
    def normalize(value: float, lower: float, upper: float) -> float:
        if not np.isfinite(value) or upper <= lower:
            return 0.0
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return default_descriptors
        
        mw = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        ring_count = Descriptors.RingCount(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)

        lipinski_violations = sum((mw > 500, logp > 5, hbd > 5, hba > 10))
        ring_atoms = sum(1 for atom in mol.GetAtoms() if atom.IsInRing())
        flexibility = rot_bonds / max(heavy_atoms, 1)
        rigidity = ring_atoms / max(mol.GetNumAtoms(), 1)

        raw_descriptors = [
            (mw, 0.0, 1000.0),
            (Descriptors.Chi1v(mol), 0.0, 20.0),
            (Descriptors.BalabanJ(mol), 0.0, 20.0),
            (Descriptors.MolMR(mol), 0.0, 150.0),
            (Descriptors.Chi0v(mol), 0.0, 30.0),
            (Descriptors.Chi2v(mol), 0.0, 15.0),
            (rot_bonds, 0.0, 15.0),
            (ring_count, 0.0, 10.0),
            (Descriptors.NumAromaticRings(mol), 0.0, 5.0),
            (Descriptors.NumAliphaticRings(mol), 0.0, 5.0),
            (Descriptors.FractionCSP3(mol), 0.0, 1.0),
            (heavy_atoms, 0.0, 50.0),
            (Descriptors.NumHeteroatoms(mol), 0.0, 20.0),
            (Descriptors.NumSaturatedRings(mol), 0.0, 5.0),
            (rdMolDescriptors.CalcNumHeterocycles(mol), 0.0, 5.0),
            (1.0 if mw <= 500 else 0.0, 0.0, 1.0),
            (Descriptors.Ipc(mol), 0.0, 1000.0),
            (Descriptors.Chi3v(mol), 0.0, 10.0),
            (Descriptors.Chi4v(mol), 0.0, 10.0),
            (lipinski_violations, 0.0, 4.0),
            (Descriptors.BertzCT(mol), 0.0, 2000.0),
            (Descriptors.LabuteASA(mol), 0.0, 300.0),
            (Descriptors.NumValenceElectrons(mol), 0.0, 200.0),
            (Descriptors.NumRadicalElectrons(mol), 0.0, 5.0),
            (Descriptors.MaxPartialCharge(mol), -1.0, 1.0),
            (Descriptors.MinPartialCharge(mol), -1.0, 1.0),
            (Descriptors.HallKierAlpha(mol), -5.0, 5.0),
            (Descriptors.Kappa1(mol), 0.0, 30.0),
            (flexibility, 0.0, 1.0),
            (rigidity, 0.0, 1.0),
        ]

        descriptors = np.asarray(
            [normalize(value, lower, upper) for value, lower, upper in raw_descriptors],
            dtype=np.float32,
        )
        if descriptors.shape != (30,):
            raise ValueError(f"Unexpected descriptor dimension: {descriptors.shape}; expected (30,)")
        return descriptors
        
    except Exception as e:
        logger.warning(f"Descriptor calculation failed for {smiles}: {str(e)}")
        return default_descriptors


def smiles_to_fingerprints(smiles):
   
    try:
        from data.pubchem_fp import GetPubChemFPs 
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Unable to parse SMILES: {smiles}; returning a zero vector")
            return {
                'morgan': np.zeros(2048, dtype=np.float32),
                'maccs': np.zeros(167, dtype=np.float32),
                'pubchem': np.zeros(881, dtype=np.float32)
            }
            
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_np = np.zeros(2048, dtype=np.float32) 
        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_np)
        
        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            
            maccs_np = np.zeros(167, dtype=np.float32) 
          
            DataStructs.ConvertToNumpyArray(maccs_fp, maccs_np)
        except Exception as e_maccs: 
            logger.warning(f"MACCS fingerprint calculation failed for {smiles}: {e_maccs}; returning a zero vector")
            maccs_np = np.zeros(167, dtype=np.float32) 
        
        try:
            mol_with_H = Chem.AddHs(mol)
            pubchem_np = GetPubChemFPs(mol_with_H) 
            if not isinstance(pubchem_np, np.ndarray) or pubchem_np.shape[0] != 881:
                logger.warning(f"PubChem fingerprint returned an invalid type or dimension ({type(pubchem_np)}, {pubchem_np.shape if hasattr(pubchem_np, 'shape') else 'N/A'}); returning a zero vector")
                pubchem_np = np.zeros(881, dtype=np.float32)
            else:
                 
                 pubchem_np = pubchem_np.astype(np.float32)
            
        except Exception as e_pubchem: 
            logger.warning(f"PubChem fingerprint calculation failed for {smiles}: {str(e_pubchem)}; returning a zero vector")
            pubchem_np = np.zeros(881, dtype=np.float32) 
        return {
            'morgan': morgan_np, 
            'maccs': maccs_np,   
            'pubchem': pubchem_np
        }
    except Exception as e:
        logger.warning(f"SMILES processing failed for {smiles}: {str(e)}; returning a zero vector")
        return {
            'morgan': np.zeros(2048, dtype=np.float32),
            'maccs': np.zeros(167, dtype=np.float32),
            'pubchem': np.zeros(881, dtype=np.float32)
        }
HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0, Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2, Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4, Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6, Chem.rdchem.HybridizationType.OTHER: 7
}
CHIRAL_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2, Chem.rdchem.ChiralType.CHI_OTHER: 3
}
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.UNSPECIFIED: 0, Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 12,
    
}

def smiles_to_graph(smiles, max_sp_dist=5):

    try:
       
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"smiles_to_graph: RDKit MolFromSmiles failed for: {smiles}. Returning None.")
            return None 

       
        try:
            mol = Chem.AddHs(mol)
            
        except Exception as e_addhs:
             logger.warning(f"smiles_to_graph: RDKit AddHs failed for {smiles}: {e_addhs}. Returning None.")
             return None

        num_nodes = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
      
        
        if num_nodes == 0:
            logger.warning(f"smiles_to_graph: Molecule has 0 atoms after AddHs: {smiles}. Returning None.")
            return None 

        
        atom_features = []
        FIXED_NODE_FEATURE_DIM = 9
        for atom in mol.GetAtoms():
             features = [
                 float(atom.GetAtomicNum()), float(atom.GetDegree()), float(atom.GetFormalCharge()),
                 float(CHIRAL_MAP.get(atom.GetChiralTag(), CHIRAL_MAP[Chem.rdchem.ChiralType.CHI_UNSPECIFIED])),
                 float(HYBRIDIZATION_MAP.get(atom.GetHybridization(), HYBRIDIZATION_MAP[Chem.rdchem.HybridizationType.UNSPECIFIED])),
                 float(atom.GetIsAromatic()), float(atom.GetTotalNumHs()), float(atom.IsInRing()), float(atom.GetMass())
             ]
             if len(features) < FIXED_NODE_FEATURE_DIM:
                 features.extend([0.0] * (FIXED_NODE_FEATURE_DIM - len(features)))
             elif len(features) > FIXED_NODE_FEATURE_DIM:
                 features = features[:FIXED_NODE_FEATURE_DIM]
             atom_features.append(features)
        try:
                x = torch.tensor(atom_features, dtype=torch.float)
                if x.dim() != 2 or x.shape[1] != FIXED_NODE_FEATURE_DIM:
                   logger.error(f"smiles_to_graph: Node feature tensor dim error for {smiles}. Expected [*, {FIXED_NODE_FEATURE_DIM}], got {x.shape}. Returning None.")
                   return None
                
                x = x.contiguous()
        except Exception as e_tensor_x:
             logger.error(f"smiles_to_graph: Error creating node feature tensor for {smiles}: {e_tensor_x}. Returning None.", exc_info=True)
             return None

        edge_indices = []
        edge_features = []
        FIXED_EDGE_FEATURE_DIM = 3
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < j:
                bond_type_enum = bond.GetBondType()
                features = [
                     float(BOND_TYPE_MAP.get(bond_type_enum, BOND_TYPE_MAP.get(Chem.rdchem.BondType.OTHER, 0))),
                     float(bond.GetIsConjugated()),
                     float(bond.IsInRing())
                ]
                if len(features) < FIXED_EDGE_FEATURE_DIM:
                    features.extend([0.0] * (FIXED_EDGE_FEATURE_DIM - len(features)))
                elif len(features) > FIXED_EDGE_FEATURE_DIM:
                    features = features[:FIXED_EDGE_FEATURE_DIM]
                edge_indices.append([i, j])
                edge_features.append(features)

        if edge_indices:
            edge_index_undirected = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
            try:
                bidirectional_features = edge_features + edge_features
                edge_attr_tensor = torch.tensor(bidirectional_features, dtype=torch.float)
                if edge_attr_tensor.dim() != 2 or edge_attr_tensor.shape[1] != FIXED_EDGE_FEATURE_DIM:
                     logger.error(f"smiles_to_graph: Edge feature tensor dim error for {smiles}. Expected [*, {FIXED_EDGE_FEATURE_DIM}], got {edge_attr_tensor.shape}. Returning None.")
                     return None
                edge_attr = edge_attr_tensor.contiguous()
            except Exception as e_tensor_edge:
                logger.error(f"smiles_to_graph: Error creating edge feature tensor for {smiles}: {e_tensor_edge}. Returning None.", exc_info=True)
                return None
        else: 
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, FIXED_EDGE_FEATURE_DIM), dtype=torch.float)
           


        in_degree = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
        out_degree = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)

        spatial_pos = None 
        if num_nodes == 1:
            spatial_pos = torch.zeros((1, 1), dtype=torch.long)
        elif len(edge_indices) == 0:
             spatial_pos = torch.full((num_nodes, num_nodes), max_sp_dist + 1, dtype=torch.long)
             spatial_pos.fill_diagonal_(0)
        else:
            try:
               
                nx_graph = nx.Graph(edge_index_undirected.t().tolist())
                nx_graph.add_nodes_from(range(num_nodes)) 
               

                
                sp_length = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=max_sp_dist))

                spatial_pos = torch.full((num_nodes, num_nodes), max_sp_dist + 1, dtype=torch.long)
                for i in range(num_nodes):
                    spatial_pos[i, i] = 0
                    if i in sp_length:
                        for j, length in sp_length[i].items():
                            if length <= max_sp_dist:
                                spatial_pos[i, j] = length
            except Exception as e_nx:
                logger.error(f"smiles_to_graph: Error calculating spatial_pos using NetworkX for {smiles}: {e_nx}. Returning None.", exc_info=True)
                return None

        if spatial_pos is None:
             logger.error(f"smiles_to_graph: spatial_pos is still None after calculation attempts for {smiles}. Returning None.")
             return None
        if spatial_pos.shape != (num_nodes, num_nodes):
             logger.error(f"smiles_to_graph: spatial_pos shape mismatch for {smiles}. Expected ({num_nodes},{num_nodes}), got {spatial_pos.shape}. Returning None.")
             return None

        x = x.contiguous()
        edge_index = edge_index.contiguous()
        edge_attr = edge_attr.contiguous()
        in_degree = in_degree.contiguous()
        out_degree = out_degree.contiguous()
        spatial_pos = spatial_pos.contiguous()

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            
        )
       
        return data

    except Exception as e:
        logger.exception(f"smiles_to_graph: Unexpected top-level error processing {smiles}: {e}. Returning None.")
        return None 

def clean_dataset_directories(dataset_name):
    raw_dir = os.path.join(RAW_DATA_DIR, dataset_name)
    processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
    
    if os.path.exists(raw_dir):
        logger.info(f"Removing raw data directory for dataset {dataset_name}")
        shutil.rmtree(raw_dir)
    
    if os.path.exists(processed_dir):
        logger.info(f"Removing processed data directory for dataset {dataset_name}")
        shutil.rmtree(processed_dir)
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(processed_dir, "graph_data", split), exist_ok=True)

def process_all_datasets(tokenizer, clean=False, default_max_seq_length=128):
    log_path = os.path.join(DATA_DIR, 'processing_log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)

    datasets_to_process_with_config = {}
    all_dataset_names = CLASSIFICATION_DATASETS + REGRESSION_DATASETS

    for name in all_dataset_names:
        task_type = 'classification' if name in CLASSIFICATION_DATASETS else 'regression'
        current_overlap_ratio = 0.0

        datasets_to_process_with_config[name] = {
            'type': task_type,
            'scaffold_overlap_ratio': current_overlap_ratio,
        }

    for dataset_name, ds_config in datasets_to_process_with_config.items():
        logger.info(f"===== Starting dataset processing: {dataset_name} ({ds_config['type']}), using a fixed scaffold split =====")
        if clean:
            clean_dataset_directories(dataset_name)
        process_dataset(
            dataset_name, 
            ds_config['type'], 
            tokenizer, 
            default_max_seq_length=default_max_seq_length,
            split_type='scaffold',
            scaffold_overlap_ratio=ds_config['scaffold_overlap_ratio'],
        )
        logger.info(f"===== Finished dataset processing: {dataset_name} =====\\n")

def process_dataset(dataset_name, task_type='classification', tokenizer=None, save_format='csv', max_seq_length=128, default_max_seq_length=128, split_type='scaffold', scaffold_overlap_ratio=0.0, dump_split_stats=False):
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )
    
    split_type = 'scaffold'
    logger.info(f"--- Starting dataset processing: {dataset_name} (task_type: {task_type}) ---")
    
    
    split_params = {
        'train_frac': 0.8,
        'valid_frac': 0.1,
        'test_frac': 0.1,
    }

    
    train_indices, valid_indices, test_indices = None, None, None

    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    config_file_path = os.path.join(project_root_dir, 'configs', f"{dataset_name.lower()}.yaml")
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config_max_seq_length = config.get('data', {}).get('max_seq_length', default_max_seq_length)
            if max_seq_length == 128:  
                max_seq_length = config_max_seq_length
                logger.info(f"Loaded max_seq_length from configuration: {max_seq_length}")
           
            enable_3d_conformers = config.get('data', {}).get('enable_3d_conformers', True)
            logger.info(f"Loaded enable_3d_conformers from configuration: {enable_3d_conformers}")
        except Exception as e_conf:
            logger.warning(f"Unable to load or parse configuration file {config_file_path}: {e_conf}. Using default values.")
            enable_3d_conformers = True 
    else:
        enable_3d_conformers = True  

    processed_dataset_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
    os.makedirs(processed_dataset_dir, exist_ok=True)
    
    df_raw = None 
    smiles_col = None
    tasks_list = []   

    logger.info(f"Trying to load dataset {dataset_name} with DeepChem...")
    loader_function = get_loader_function_for_dataset(dataset_name)

    if loader_function:
        try:
            dc_tasks, dc_datasets, dc_transformers = loader_function(
                splitter=None, reload=False, transformers=[]
            )

            if not dc_datasets:
                raise ValueError("The DeepChem loader did not return a dataset tuple.")

            all_dfs = []
            for i, d_set in enumerate(dc_datasets):
                if d_set is None:
                    logger.warning(f"DeepChem returned None for dataset part {i}.")
                    continue

                logger.info(f"DeepChem returned dataset part {i} with {len(d_set)} samples.")
                if isinstance(d_set, dc.data.Dataset):
                    all_dfs.append(d_set.to_dataframe())
                else:
                    for sub_d_set in d_set:
                        if sub_d_set is not None and isinstance(sub_d_set, dc.data.Dataset):
                            all_dfs.append(sub_d_set.to_dataframe())

            if not all_dfs:
                raise ValueError("The DataFrame list converted from DeepChem datasets is empty.")

            df_raw = pd.concat(all_dfs, ignore_index=True)
            total_samples = len(df_raw)
            logger.info(
                f"Successfully loaded and merged {total_samples} samples "
                f"from {dataset_name} with DeepChem."
            )

            tasks_list = list(dc_tasks) if dc_tasks else []
            logger.info(f"Tasks reported by DeepChem: {tasks_list}")

            if 'ids' in df_raw.columns:
                smiles_col = 'ids'
            elif 'X' in df_raw.columns:
                if (
                    len(df_raw) > 0
                    and isinstance(df_raw['X'].iloc[0], str)
                    and Chem.MolFromSmiles(df_raw['X'].iloc[0]) is not None
                ):
                    smiles_col = 'X'
            elif 'smiles' in df_raw.columns:
                smiles_col = 'smiles'

            if smiles_col:
                logger.info(
                    f"Identified the SMILES column from DeepChem data as: '{smiles_col}'"
                )
                df_raw.rename(columns={smiles_col: 'smiles'}, inplace=True)
                smiles_col = 'smiles'
            else:
                logger.error(
                    "Unable to identify the SMILES column automatically from DeepChem data. "
                    f"Columns: {df_raw.columns.tolist()}"
                )
                raise ValueError("Unable to identify the SMILES column")

            if not tasks_list:
                logger.warning(
                    "DeepChem did not report task names; inferring them from DataFrame columns."
                )

                potential_task_cols_from_df = [
                    c for c in df_raw.columns if c not in [smiles_col, 'w', 'X', 'ids']
                ]
                if 'y' in df_raw.columns:
                    if (
                        len(df_raw) > 0
                        and isinstance(df_raw['y'].iloc[0], (list, np.ndarray))
                        and len(df_raw['y'].iloc[0]) > 1
                    ):
                        num_deepchem_tasks = len(df_raw['y'].iloc[0])
                        temp_task_names = [
                            f"task_{i + 1}" for i in range(num_deepchem_tasks)
                        ]
                        y_df = pd.DataFrame(
                            df_raw['y'].to_list(), columns=temp_task_names
                        )
                        df_raw = pd.concat(
                            [df_raw.drop(columns=['y']), y_df], axis=1
                        )
                        tasks_list = temp_task_names
                        logger.info(
                            "DeepChem returned a multidimensional 'y' column; "
                            f"split it into tasks: {tasks_list}"
                        )
                    else:
                        tasks_list = ['y']
                elif potential_task_cols_from_df:
                    tasks_list = potential_task_cols_from_df
                else:
                    raise ValueError(
                        "Unable to determine task columns from DeepChem data."
                    )
                logger.info(f"Tasks inferred from the DataFrame: {tasks_list}")
            else:
                missing_dc_tasks_in_df = [
                    t for t in tasks_list if t not in df_raw.columns
                ]

                if not missing_dc_tasks_in_df:
                    logger.info(
                        "The tasks reported by DeepChem already exist as DataFrame "
                        f"columns: {tasks_list}."
                    )
                else:
                    mapped_successfully = False
                    if 'y' in df_raw.columns:
                        logger.info(
                            f"Attempting to map DeepChem tasks {tasks_list} to the 'y' column."
                        )
                        if (
                            len(tasks_list) == 1
                            and tasks_list[0] not in df_raw.columns
                        ):
                            df_raw.rename(
                                columns={'y': tasks_list[0]}, inplace=True
                            )
                            logger.info(
                                f"Renamed the 'y' column to task '{tasks_list[0]}'."
                            )
                            mapped_successfully = True
                        elif (
                            len(df_raw) > 0
                            and isinstance(
                                df_raw['y'].iloc[0], (list, np.ndarray)
                            )
                            and len(df_raw['y'].iloc[0]) == len(tasks_list)
                        ):
                            try:
                                if any(
                                    task_name in df_raw.columns
                                    for task_name in tasks_list
                                ):
                                    logger.warning(
                                        "Some target task names already exist as DataFrame "
                                        f"columns while splitting 'y': {tasks_list}; "
                                        "skipping the split to avoid overwriting data."
                                    )
                                else:
                                    y_df = pd.DataFrame(
                                        df_raw['y'].to_list(), columns=tasks_list
                                    )
                                    df_raw = pd.concat(
                                        [df_raw.drop(columns=['y']), y_df], axis=1
                                    )
                                    logger.info(
                                        "Successfully split the 'y' column into tasks: "
                                        f"{tasks_list}"
                                    )
                                    mapped_successfully = True
                            except Exception as e_split_y:
                                logger.warning(
                                    f"Failed to split the 'y' column: {e_split_y}. "
                                    "Trying another strategy."
                                )
                        else:
                            logger.info(
                                "The 'y' column exists but its structure does not "
                                f"match the task list {tasks_list}."
                            )

                    if not mapped_successfully and tasks_list:
                        logger.info(
                            f"Attempting to map DeepChem tasks {tasks_list} "
                            "to 'y1', 'y2', ... columns."
                        )
                        potential_y_cols_map = {
                            f'y{i + 1}': tasks_list[i]
                            for i in range(len(tasks_list))
                        }
                        actual_y_cols_to_rename = {
                            k: v
                            for k, v in potential_y_cols_map.items()
                            if k in df_raw.columns and v not in df_raw.columns
                        }

                        if (
                            len(actual_y_cols_to_rename) == len(tasks_list)
                            and all(
                                f'y{i + 1}' in df_raw.columns
                                for i in range(len(tasks_list))
                            )
                        ):
                            df_raw.rename(
                                columns=actual_y_cols_to_rename, inplace=True
                            )
                            logger.info(
                                f"Renamed {list(actual_y_cols_to_rename.keys())} "
                                f"to tasks: {list(actual_y_cols_to_rename.values())}"
                            )
                            mapped_successfully = True

                            potential_w_cols_map = {
                                f'w{i + 1}': f"w_{tasks_list[i]}"
                                for i in range(len(tasks_list))
                            }
                            actual_w_cols_to_rename_map = {
                                k: v
                                for k, v in potential_w_cols_map.items()
                                if k in df_raw.columns and v not in df_raw.columns
                            }
                            if actual_w_cols_to_rename_map:
                                df_raw.rename(
                                    columns=actual_w_cols_to_rename_map,
                                    inplace=True,
                                )
                                logger.info(
                                    "Renamed weight columns "
                                    f"{list(actual_w_cols_to_rename_map.keys())} "
                                    f"to {list(actual_w_cols_to_rename_map.values())}"
                                )
                        elif len(actual_y_cols_to_rename) > 0:
                            logger.warning(
                                "Only some 'y1', 'y2', ... columns were found; "
                                f"they do not cover all tasks {tasks_list} or contain "
                                f"conflicts. Expected: {len(tasks_list)}; safely "
                                f"renameable: {len(actual_y_cols_to_rename)}."
                            )
                        else:
                            logger.info(
                                "Could not find 'y1', 'y2', ... columns for task "
                                f"list {tasks_list}, or the target column names "
                                "already exist."
                            )

                    if not mapped_successfully:
                        logger.warning(
                            "Unable to map the tasks reported by DeepChem "
                            f"{tasks_list} (missing from df_raw: "
                            f"{missing_dc_tasks_in_df}) to DataFrame columns. "
                            f"df_raw columns: {df_raw.columns.tolist()}"
                        )

            final_tasks_in_df = [
                t for t in tasks_list if t in df_raw.columns
            ]
            if (
                len(final_tasks_in_df) != len(tasks_list)
                and tasks_list
            ):
                logger.warning(
                    "The original DeepChem task list "
                    f"{tasks_list} does not fully match the processed DataFrame "
                    f"task columns {final_tasks_in_df}; using the latter."
                )
            tasks_list = final_tasks_in_df

            cols_to_drop_from_dc = ['w']
            if 'X' in df_raw.columns and smiles_col != 'X':
                cols_to_drop_from_dc.append('X')
            df_raw.drop(
                columns=cols_to_drop_from_dc, errors='ignore', inplace=True
            )
            logger.info(
                f"DeepChem data columns after processing: {df_raw.columns.tolist()}"
            )

        except Exception as e_dc:
            logger.error(
                f"Failed to load dataset {dataset_name} with DeepChem: {e_dc}"
            )
            logger.info(
                "Falling back to manual local CSV/CSV.GZ discovery and processing..."
            )
            df_raw = None
            smiles_col = None
            tasks_list = []
    else:
        logger.warning(
            f"No DeepChem loader found for {dataset_name}; "
            "trying local CSV/CSV.GZ files..."
        )
        df_raw = None
        smiles_col = None
        tasks_list = []

    if df_raw is None: 
        logger.info(f"Searching manually for CSV/CSV.GZ files...")
        dataset_file = None
        data_dir_dc = get_data_dir()
        base_filenames = [
            f"{dataset_name.upper()}", f"{dataset_name.lower()}", f"{dataset_name}"
        ]
        dataset_file_aliases = {
            'esol': ['delaney', 'DELANEY', 'Delaney', 'delaney-processed'],
            'lipo': ['lipophilicity', 'LIPOPHILICITY', 'Lipophilicity']
        }
        if dataset_name.lower() in dataset_file_aliases:
            base_filenames.extend(dataset_file_aliases[dataset_name.lower()])
        
        possible_paths = [] 
        for fname in set(base_filenames):
            for ext in ['.csv', '.csv.gz']:
                possible_paths.append(os.path.join(data_dir_dc, fname + ext))
                possible_paths.append(os.path.join("/tmp", fname + ext))
                possible_paths.append(os.path.join(RAW_DATA_DIR, dataset_name.lower(), fname + ext))
                possible_paths.append(os.path.join(RAW_DATA_DIR, fname + ext))
        logger.info(f"Searching the following CSV/CSV.GZ paths (some may not exist): {list(set(possible_paths))}")
        for filepath in set(possible_paths):
            if os.path.exists(filepath):
                dataset_file = filepath
                logger.info(f"Found CSV/CSV.GZ dataset file: {dataset_file}")
                break

        if dataset_file is None:
            logger.error(f"DeepChem loading failed and no CSV/CSV.GZ file for {dataset_name} was found. Ensure the data is downloaded or accessible through DeepChem; skipping this dataset.")
            return

        try:
            df_raw = pd.read_csv(dataset_file)
            logger.info(f"Successfully read CSV file manually: {dataset_file}")
            logger.info(f"Raw CSV columns: {df_raw.columns.tolist()}")
        
            
            possible_smiles_cols = ['smiles', 'canonical_smiles', 'isomericsmiles', 'smiles string', 'text', 'description']
            smiles_col = None 
            for col_s in df_raw.columns:
                if col_s.lower() in possible_smiles_cols:
                    smiles_col = col_s
                    logger.info(f"Automatically detected SMILES column in manual mode: {smiles_col}")
                    break
            if smiles_col is None:
                smiles_col = df_raw.columns[0]
                logger.warning(f"Could not detect the SMILES column in manual mode; assuming the first column: {smiles_col}")
            
            df_raw.rename(columns={smiles_col: 'smiles'}, inplace=True) 
            smiles_col = 'smiles'

           
            split_params = {
                'train_frac': 0.8,
                'valid_frac': 0.1,
                'test_frac': 0.1,
            }

            potential_task_cols = [col for col in df_raw.columns if col != smiles_col]
            tasks_list = [] 

            if dataset_name_lower == 'tox21':
                tox21_tasks_candidates = [f'NR-{task}' for task in ['AR', 'AR-LBD', 'ER', 'ER-LBD', 'Aromatase', 'ATAD5', 'HSE', 'MMP', 'PPAR-gamma', 'p53', 'ARE']] 
                tox21_tasks_candidates += [f'SR-{task}' for task in ['ARE', 'ATAD5', 'ER', 'HSE', 'MMP', 'p53']] 
                tasks_list = [col for col in tox21_tasks_candidates if col in df_raw.columns]
                if not tasks_list: tasks_list = [col for col in potential_task_cols if col.startswith('NR-') or col.startswith('SR-')]
            elif dataset_name_lower == 'sider':
                binary_cols_sider = [col for col in potential_task_cols if df_raw[col].dropna().isin([0, 1, 0.0, 1.0]).all()]
                tasks_list = binary_cols_sider if binary_cols_sider else potential_task_cols
                if not tasks_list and potential_task_cols:
                    tasks_list = [potential_task_cols[0]]
            elif dataset_name_lower == 'clintox':
                clintox_tasks_candidates = ['FDA_APPROVED', 'CT_TOX']
                tasks_list = [col for col in clintox_tasks_candidates if col in df_raw.columns]
                if not tasks_list: tasks_list = potential_task_cols
            elif dataset_name_lower == 'bace':
                bace_tasks_candidates = ['Class', 'Activity', 'bace_label', 'pXC50']
                tasks_list = [col for col in bace_tasks_candidates if col in df_raw.columns]
                if not tasks_list and potential_task_cols: tasks_list = [potential_task_cols[0]]
            elif dataset_name_lower == 'bbbp':
                if 'p_np' in df_raw.columns:
                    tasks_list = ['p_np']
                elif 'Class' in df_raw.columns:
                    tasks_list = ['Class']
                else:
                    binary_cols_bbbp = [col for col in potential_task_cols if df_raw[col].dropna().isin([0, 1, 0.0, 1.0]).all()]
                    tasks_list = binary_cols_bbbp if binary_cols_bbbp else []
                if not tasks_list and potential_task_cols:
                    tasks_list = [potential_task_cols[0]]
            elif dataset_name_lower == 'esol':
                esol_tasks_candidates = ['measured log solubility in mols per litre', 'ESOL', 'logS', 'Solubility']
                tasks_list = [col for col in esol_tasks_candidates if col in df_raw.columns]
                if not tasks_list and potential_task_cols: tasks_list = [potential_task_cols[0]]
            elif dataset_name_lower == 'freesolv':
                fs_tasks_candidates = ['expt', 'calc', 'freesolv_logG', 'hydration free energy']
                tasks_list = [col for col in fs_tasks_candidates if col in df_raw.columns and col == 'expt'] 
                if not tasks_list: tasks_list = [col for col in fs_tasks_candidates if col in df_raw.columns]
                if not tasks_list and potential_task_cols: tasks_list = [potential_task_cols[0]]
            elif dataset_name_lower in ['lipo', 'lipophilicity']:
                lipo_tasks_candidates = ['exp', 'lipo', 'logD', 'Lipophilicity']
                tasks_list = [col for col in lipo_tasks_candidates if col in df_raw.columns and col == 'exp']
                if not tasks_list: tasks_list = [col for col in lipo_tasks_candidates if col in df_raw.columns]
                if not tasks_list and potential_task_cols: tasks_list = [potential_task_cols[0]]
            else: 
                if task_type == 'classification':
                    binary_cols = [col for col in potential_task_cols if df_raw[col].dropna().isin([0, 1, 0.0, 1.0]).all()]
                    tasks_list = binary_cols if binary_cols else ([potential_task_cols[0]] if potential_task_cols else [])
                elif task_type == 'regression':
                    numeric_cols = [col for col in potential_task_cols if pd.api.types.is_numeric_dtype(df_raw[col])]
                    tasks_list = numeric_cols if numeric_cols else ([potential_task_cols[0]] if potential_task_cols else [])

            if not tasks_list:
                logger.error(f"Manual mode could not determine any task columns for dataset {dataset_name}; skipping this dataset.")
                return
            logger.info(f"Final task columns in manual mode: {tasks_list} for task_type: {task_type}")

        except Exception as e_csv_manual:
            logger.error(f"Manual CSV processing failed: {e_csv_manual}")
            import traceback; logger.error(traceback.format_exc())
            return

    
    logger.info(f"=== Step 3.5: Cleaning invalid SMILES ===")
    original_sample_count = len(df_raw)
    
    df_raw, cleaned_samples = clean_invalid_smiles(
        df=df_raw,
        smiles_col='smiles',
        output_dir=processed_dataset_dir,
        dataset_name=dataset_name
    )
    
    if cleaned_samples:
        logger.warning(f"⚠️ Cleaned {len(cleaned_samples)} invalid SMILES samples")
        logger.warning(f"   Original samples: {original_sample_count} -> after cleaning: {len(df_raw)}")
    else:
        logger.info(f"✅ All SMILES are valid; no cleaning was required")
    
    total_samples = len(df_raw)

    
    filter_max_atoms = 150
    filter_label_outliers = False 
    
    if dataset_name_lower in SUPPORTED_DATASETS:
        filter_label_outliers = dataset_name_lower in REGRESSION_DATASETS
    
    df_raw, outlier_stats = filter_outliers(
        df=df_raw,
        task_type=task_type,
        tasks_list=tasks_list,
        max_atoms=filter_max_atoms,
        remove_label_outliers=filter_label_outliers
    )
    total_samples = len(df_raw) 
    if total_samples == 0:
        logger.error(f"Unable to determine any task columns for dataset {dataset_name} after cleaning and outlier filtering; stopping.")
        return
    if not tasks_list:
        logger.error(f"No valid task columns found for dataset {dataset_name}; stopping.")
        return

    logger.info("Using deterministic scaffold splitting for dataset %s.", dataset_name)
    try:
        train_indices, valid_indices, test_indices = scaffold_split(
            df_raw['smiles'].tolist(),
            train_frac=split_params['train_frac'],
            valid_frac=split_params['valid_frac'],
            test_frac=split_params['test_frac'],
        )
    except Exception as e_split:
        logger.error("Error during scaffold splitting: %s", e_split, exc_info=True)
        return

    logger.info(
        "Scaffold split completed: train=%d, validation=%d, test=%d.",
        len(train_indices),
        len(valid_indices),
        len(test_indices),
    )
    
    logger.info(f"Final split sizes: train={len(train_indices)}, validation={len(valid_indices)}, test={len(test_indices)}")
    
    assert len(set(train_indices).intersection(valid_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(valid_indices).intersection(test_indices)) == 0
    
    if len(train_indices) + len(valid_indices) + len(test_indices) != total_samples:
        logger.warning(f"Split sample count does not match the original count: {len(train_indices) + len(valid_indices) + len(test_indices)} != {total_samples}")
        logger.warning(f"Possibly ignored {total_samples - (len(train_indices) + len(valid_indices) + len(test_indices))} invalid molecules")

    logger.info(f"Dataset {dataset_name} splitting completed.")

    original_to_group_id = {
        idx: original_idx for original_idx, idx in enumerate(df_raw.index.tolist())
    }

    def build_split_data(indices, split_name):
        split_data = []
        for position in indices:
            row_index = df_raw.index[position]
            molecule_id = original_to_group_id[row_index]
            split_data.append({
                'smiles': df_raw.at[row_index, 'smiles'],
                'sample_id': f"{split_name}_{row_index}",
                'original_molecule_id_for_files': row_index,
                'original_molecule_group_id': molecule_id,
                'labels': df_raw.loc[row_index, tasks_list].tolist(),
            })
        logger.info(f"The {split_name} split contains {len(split_data)} samples.")
        return split_data

    train_data = build_split_data(train_indices, 'train')
    val_data = build_split_data(valid_indices, 'val')
    test_data = build_split_data(test_indices, 'test')

    normalization_params = None
    if task_type == 'regression':
        logger.info(f"Computing normalization parameters for {len(tasks_list)} training tasks...")
        train_df = pd.DataFrame(train_data)
        normalization_params = calculate_normalization(train_df, tasks_list)
        logger.info(f"Saving label normalization parameters to {processed_dataset_dir}/label_normalization.json")
        with open(os.path.join(processed_dataset_dir, 'label_normalization.json'), 'w') as f:
            json.dump(normalization_params, f, indent=4, cls=NumpyEncoder)

    process_and_save_split_data(pd.DataFrame(train_data), processed_dataset_dir, tokenizer, 'train', tasks_list, task_type, normalization_params, max_seq_length, dataset_name, enable_3d_conformers)
    process_and_save_split_data(pd.DataFrame(val_data), processed_dataset_dir, tokenizer, 'val', tasks_list, task_type, normalization_params, max_seq_length, dataset_name, enable_3d_conformers)
    process_and_save_split_data(pd.DataFrame(test_data), processed_dataset_dir, tokenizer, 'test', tasks_list, task_type, normalization_params, max_seq_length, dataset_name, enable_3d_conformers)

    create_dataset_info_file(dataset_name, tasks_list, task_type)
    
    if task_type == 'classification':
        try:
            split_stats = {}
            for task_name in tasks_list:
                split_stats[task_name] = {
                    'train': {
                        'total': len(train_data),
                        'positive': 0,
                        'negative': 0
                    },
                    'val': {
                        'total': len(val_data),
                        'positive': 0,
                        'negative': 0
                    },
                    'test': {
                        'total': len(test_data),
                        'positive': 0,
                        'negative': 0
                    }
                }
            
            for task_idx, task_name in enumerate(tasks_list):
                for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                    for sample in split_data:
                        labels = sample.get('labels', [])
                        if len(labels) > task_idx:
                            label = labels[task_idx]
                            if label is not None and not (isinstance(label, float) and np.isnan(label)):
                                if label > 0.5:  
                                    split_stats[task_name][split_name]['positive'] += 1
                                else:
                                    split_stats[task_name][split_name]['negative'] += 1
                
           
            split_stats['_metadata'] = {
                'original_total_samples': original_sample_count,
                'cleaned_total_samples': total_samples,
                'removed_invalid_smiles': len(cleaned_samples) if cleaned_samples else 0,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                'cleaning_applied': len(cleaned_samples) > 0 if cleaned_samples else False,
                'split_type': split_type,
                'scaffold_overlap_ratio': scaffold_overlap_ratio,
                'split_fracs': {
                    'train_frac': float(split_params.get('train_frac', 0.8)),
                    'valid_frac': float(split_params.get('valid_frac', 0.1)),
                    'test_frac': float(split_params.get('test_frac', 0.1)),
                }
            }

            try:
                idx_blob = json.dumps(
                    {
                        'train': sorted([int(x) for x in train_indices]),
                        'val': sorted([int(x) for x in valid_indices]),
                        'test': sorted([int(x) for x in test_indices]),
                    },
                    separators=(',', ':'),
                    ensure_ascii=False
                ).encode('utf-8')
                split_stats['_metadata']['indices_sha1'] = hashlib.sha1(idx_blob).hexdigest()
            except Exception as e_hash:
                logger.debug(f"Failed to compute indices_sha1 (non-fatal): {e_hash}")
            
            stats_file = os.path.join(processed_dataset_dir, 'split_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(split_stats, f, indent=4)
            logger.info(f"✅ Saved dataset statistics to {stats_file}")
            first_task = tasks_list[0]
            logger.info(f"   Train: {split_stats[first_task]['train']['positive']} positive, {split_stats[first_task]['train']['negative']} negative")
            logger.info(f"   Val: {split_stats[first_task]['val']['positive']} positive, {split_stats[first_task]['val']['negative']} negative")
            logger.info(f"   Test: {split_stats[first_task]['test']['positive']} positive, {split_stats[first_task]['test']['negative']} negative")
            if split_stats['_metadata']['removed_invalid_smiles'] > 0:
                logger.info(f"   ⚠️ Cleaned invalid SMILES: {split_stats['_metadata']['removed_invalid_smiles']}")
        except Exception as e_stats:
            logger.warning(f"Failed to generate split_stats.json: {e_stats}")
    else:
        try:
            split_stats = {
                '_metadata': {
                    'original_total_samples': original_sample_count,
                    'cleaned_total_samples': total_samples,
                    'removed_invalid_smiles': len(cleaned_samples) if cleaned_samples else 0,
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data),
                    'cleaning_applied': len(cleaned_samples) > 0 if cleaned_samples else False,
                    'task_type': 'regression',
                    'split_type': split_type,
                    'scaffold_overlap_ratio': scaffold_overlap_ratio,
                    'split_fracs': {
                        'train_frac': float(split_params.get('train_frac', 0.8)),
                        'valid_frac': float(split_params.get('valid_frac', 0.1)),
                        'test_frac': float(split_params.get('test_frac', 0.1)),
                    }
                }
            }

            try:
                idx_blob = json.dumps(
                    {
                        'train': sorted([int(x) for x in train_indices]),
                        'val': sorted([int(x) for x in valid_indices]),
                        'test': sorted([int(x) for x in test_indices]),
                    },
                    separators=(',', ':'),
                    ensure_ascii=False
                ).encode('utf-8')
                split_stats['_metadata']['indices_sha1'] = hashlib.sha1(idx_blob).hexdigest()
            except Exception as e_hash:
                logger.debug(f"Failed to compute indices_sha1 (non-fatal): {e_hash}")
            stats_file = os.path.join(processed_dataset_dir, 'split_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(split_stats, f, indent=4)
            logger.info(f"✅ Saved dataset statistics to {stats_file}")
            logger.info(f"   Train: {len(train_data)} samples, Val: {len(val_data)} samples, Test: {len(test_data)} samples")
            if split_stats['_metadata']['removed_invalid_smiles'] > 0:
                logger.info(f"   ⚠️ Cleaned invalid SMILES: {split_stats['_metadata']['removed_invalid_smiles']}")
        except Exception as e_stats:
            logger.warning(f"Failed to generate split_stats.json: {e_stats}")

    logger.info(f"--- Finished processing dataset: {dataset_name} (task_type: {task_type}) ---")

def calculate_normalization(df, task_columns):
    normalization_params = {}
    logger.info(
        f"Computing normalization parameters for {len(task_columns)} training tasks..."
    )

    if not task_columns:
        return normalization_params

    labels_column = df.get('labels') if 'labels' in df.columns else None
    for task_idx, task in enumerate(task_columns):
        try:
            if labels_column is not None:
                values = []
                for label in labels_column:
                    if isinstance(label, (list, tuple, np.ndarray)):
                        values.append(
                            label[task_idx] if task_idx < len(label) else np.nan
                        )
                    else:
                        values.append(label if task_idx == 0 else np.nan)
                labels = pd.to_numeric(
                    pd.Series(values), errors='coerce'
                ).dropna()
            elif task in df.columns:
                labels = pd.to_numeric(
                    df[task], errors='coerce'
                ).dropna()
            else:
                labels = pd.Series(dtype=np.float64)

            if labels.empty:
                logger.warning(
                    f"Task '{task}' has no valid numeric labels; "
                    "using mean=0 and std=1."
                )
                normalization_params[task] = {'mean': 0.0, 'std': 1.0}
                continue

            mean = float(labels.mean())
            std = float(labels.std())
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
                logger.warning(
                    f"Task '{task}' has an invalid or near-zero standard "
                    "deviation; setting it to 1.0."
                )

            normalization_params[task] = {'mean': mean, 'std': std}
            logger.info(
                f"Task '{task}': mean={mean:.4f}, std={std:.4f}"
            )
        except Exception as exc:
            logger.error(
                f"Failed to compute normalization parameters for task "
                f"'{task}': {exc}"
            )
            normalization_params[task] = {'mean': 0.0, 'std': 1.0}

    return normalization_params

def process_and_save_split_data(
    df: pd.DataFrame, 
    processed_dir: str, 
    tokenizer, 
    split_name: str, 
    tasks: List[str],
    task_type: str,
    normalization_params: Optional[Dict],
    max_seq_length: int = 128,
    dataset_name: str = "UNKNOWN",  
    enable_3d_conformers: bool = True  
):
    

    logger.info(f"Adding features to {len(df)} {split_name} molecules; tasks={tasks}, type={task_type}, max sequence length={max_seq_length}")

    fingerprint_dir = os.path.join(processed_dir, "fingerprints", split_name)
    graph_dir = os.path.join(processed_dir, "graph_data", split_name)
    conformer_dir = os.path.join(processed_dir, "conformers")  
    token_dir = None
    if tokenizer:
        token_dir = os.path.join(processed_dir, "token_data", split_name)
        os.makedirs(token_dir, exist_ok=True)
    os.makedirs(fingerprint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(conformer_dir, exist_ok=True) 

    graphormer_max_sp_dist = 5

    processed_indices = []
    processed_labels_list = []
    processed_smiles = []
    processed_original_molecule_group_ids_list = []
    processed_sample_ids = []
    processed_augmented_sample_ids = processed_sample_ids

    original_df_indices = df.index

    saved_original_mol_graphs = set()
    saved_original_mol_fingerprints = set()
    saved_original_mol_conformers = set() 

    logger.info(
        "3D conformer pipeline: ETKDGv3 -> MMFF94s (maxIters=%d); "
        "retaining converged, non-converged, and parameter-unavailable conformers.",
        CONFORMER_MAX_ITERS,
    ) if enable_3d_conformers else logger.info("3D conformer generation is disabled.")

    for sample_index in tqdm(original_df_indices, total=len(df),
                             desc=f"Processing {split_name} samples", unit="sample"):
        row = df.loc[sample_index]

        if pd.isna(row['smiles']) or not isinstance(row['smiles'], str) or row['smiles'].strip() == '':
            logger.warning(f"Sample {sample_index} has an empty or invalid SMILES string: '{row['smiles']}'. Skipping.")
            continue

        smiles = row['smiles']
        original_molecule_group_id = row['original_molecule_group_id']
        
        augmented_sample_id = row.get('augmented_sample_id', f"{split_name}_{sample_index}_0")

        try:
           
            graph_ready = original_molecule_group_id in saved_original_mol_graphs
            if not graph_ready:
                pyg_data = smiles_to_graph(smiles, max_sp_dist=graphormer_max_sp_dist)
                if pyg_data is not None:
                    graphormer_dict = convert_pyg_to_graphormer_dict(pyg_data)
                    combined_graph_data = {'pyg_data': pyg_data, **graphormer_dict}
                    graph_file_path = os.path.join(graph_dir, f"{original_molecule_group_id}.pt")
                    torch.save(combined_graph_data, graph_file_path)
                    saved_original_mol_graphs.add(original_molecule_group_id)
                    graph_ready = True
                else:
                    logger.warning(f"Graph generation failed for sample {sample_index}; skipping this sample.")

          
            fingerprint_ready = original_molecule_group_id in saved_original_mol_fingerprints
            if not fingerprint_ready:
                fingerprints = smiles_to_fingerprints(smiles)
                if fingerprints is not None:
                    fingerprint_file_path = os.path.join(fingerprint_dir, f"{original_molecule_group_id}.npz")
                    concatenated_fp = np.concatenate([
                        fingerprints['maccs'], fingerprints['pubchem'], fingerprints['morgan']
                    ])
                    np.savez_compressed(fingerprint_file_path, fingerprints=concatenated_fp)
                    saved_original_mol_fingerprints.add(original_molecule_group_id)
                    fingerprint_ready = True
                else:
                    logger.warning(f"Fingerprint generation failed for sample {sample_index}; skipping this sample.")

            if not graph_ready or not fingerprint_ready:
                continue

           
            conformer_ready = original_molecule_group_id in saved_original_mol_conformers
            if enable_3d_conformers and not conformer_ready:
                try:
                    conformer_result = generate_single_conformer(
                        smiles, seed=42 + int(original_molecule_group_id)
                    )
                    if conformer_result is not None:
                        conformer_result = add_pyg_data_to_conformers(conformer_result)
                        conformer_file_path = os.path.join(
                            conformer_dir, f"{original_molecule_group_id}.pt"
                        )
                        torch.save(conformer_result, conformer_file_path)
                        saved_original_mol_conformers.add(original_molecule_group_id)
                        conformer_ready = True
                        logger.debug(
                            "Molecule %s: MMFF94s status=%s",
                            original_molecule_group_id,
                            conformer_result.get('mmff_status', 'unknown'),
                        )
                    else:
                        logger.warning(
                            "Molecule %s: all ETKDGv3 embedding attempts failed; excluding the sample.",
                            original_molecule_group_id,
                        )
                except Exception as e:
                    logger.error(
                        "Molecule %s: conformer processing failed; excluding the sample: %s",
                        original_molecule_group_id,
                        e,
                    )

           
            if enable_3d_conformers and not conformer_ready:
                continue

            raw_label_list = []
            
            if 'labels' in row and isinstance(row['labels'], list) and len(row['labels']) > 0:
                
                raw_label_list = row['labels']
            else:
                common_label_keys = ['label', 'labels', 'y']
                
                possible_label_keys = common_label_keys
                
                label_key = None
                for key in possible_label_keys:
                    if key in row and not pd.isna(row[key]):
                        label_key = key
                        break
                
               
                if label_key:
                    if isinstance(row[label_key], list):
                        raw_label_list = row[label_key]
                    else:
                        raw_label_list = [row[label_key]]
                else:
                  
                    task_found = False
                    for task in tasks:
                        if task in row and not pd.isna(row[task]):
                            if not task_found: 
                                raw_label_list = []
                                task_found = True
                            raw_label_list.append(row[task])
                    
                   
                    if not task_found:
                        logger.warning(f"Sample {sample_index} is missing labels or contains NaN labels.")
                        raw_label_list = [float('nan')] * len(tasks)

            processed_labels = []
            for task_idx, task in enumerate(tasks):
               
                task_raw_val = raw_label_list[task_idx] if task_idx < len(raw_label_list) else float('nan')

                processed_label = float('nan')  

                if task_type == 'classification':
                    try:
                        label_float = float(task_raw_val)
                        label_int = int(label_float)
                        if label_float == label_int and label_int in [0, 1]:
                            processed_label = float(label_int)
                    except (ValueError, TypeError):
                        pass  
              
                elif task_type == 'regression':
                    try:
                        if pd.isna(task_raw_val):
                            pass  
                        else:
                            label_float = float(task_raw_val)
                            if normalization_params and task in normalization_params:
                                mean = normalization_params[task].get('mean', 0.0)
                                std = normalization_params[task].get('std', 1.0)
                             
                                std = std if std > 1e-6 else 1.0
                                processed_label = (label_float - mean) / std
                            else:
                                processed_label = label_float  
                    except (ValueError, TypeError):
                        logger.warning(f"Task '{task}', sample {sample_index}: unable to convert label value '{task_raw_val}' to float; setting it to NaN.")
                        pass
                else:
                    processed_label = task_raw_val  

                processed_labels.append(processed_label)

            processed_indices.append(sample_index)
            processed_labels_list.append(processed_labels)
            processed_smiles.append(smiles)
            processed_original_molecule_group_ids_list.append(original_molecule_group_id)
            processed_augmented_sample_ids.append(augmented_sample_id)  

        except Exception as e:
            logger.error(f"Error while processing sample {sample_index}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully processed {len(processed_indices)} / {len(df)} {split_name} samples.")
    
    if enable_3d_conformers:
        logger.info(f"Generated 3D conformers for {len(saved_original_mol_conformers)} unique molecules.")
        print_conformer_generation_stats()

    if len(processed_indices) == 0:
        logger.warning(f"No samples were successfully processed in the {split_name} split; skipping index-file saving.")
        return

    split_index_data = {
        'split': split_name,
        'num_samples': len(processed_indices),
        'smiles': processed_smiles,
        'labels': processed_labels_list,
        'idx_mapping': {str(i): processed_augmented_sample_ids[i] for i in range(len(processed_augmented_sample_ids))}, 
        'original_molecule_group_ids': processed_original_molecule_group_ids_list
    }

    index_file_path = os.path.join(processed_dir, f"{split_name}_indices.json")
    with open(index_file_path, 'w') as f:
        json.dump(split_index_data, f, indent=4, cls=NumpyEncoder)

    logger.info(f"Finished feature processing and saving for the {split_name} split: {len(processed_indices)} samples.")


def add_pyg_data_to_conformers(conformer_result: Dict[str, Any]) -> Dict[str, Any]:
    """Add a PyG data object to conformer data for SchNet and other 3D models."""
    if not conformer_result['success'] or not conformer_result['conformers']:
        return conformer_result
    
    try:
        best_conformer = conformer_result['conformers'][0]
        
        if 'coordinates' in best_conformer and 'atomic_numbers' in best_conformer:
            coords = best_conformer['coordinates']
            atomic_nums = best_conformer['atomic_numbers']
            
        
            if isinstance(coords, np.ndarray):
                pos = torch.from_numpy(coords).float()
            else:
                pos = torch.tensor(coords).float()
                
            if isinstance(atomic_nums, (list, np.ndarray)):
                z = torch.tensor(atomic_nums).long()
            else:
                z = atomic_nums.long()
            
            num_atoms = min(len(atomic_nums), pos.size(0))
            if pos.size(0) != num_atoms:
                logger.warning(f"Coordinate count ({pos.size(0)}) does not match atom count ({num_atoms}); truncating to {num_atoms}.")
                pos = pos[:num_atoms]
                z = z[:num_atoms]
            
            
            x = torch.zeros((num_atoms, 9), dtype=torch.float)
            
            x[:, 0] = z.float()
            
            if 'atomic_features' in best_conformer:
                atomic_features = best_conformer['atomic_features']
                if isinstance(atomic_features, list) and len(atomic_features) >= num_atoms:
                    for i in range(num_atoms):
                        features = atomic_features[i]
                        if isinstance(features, (list, tuple, np.ndarray)) and len(features) > 0:
                            feat_len = min(len(features), 8)  
                            if feat_len > 0:
                                try:
                                    if isinstance(features[0], (int, float, np.number)):
                                        x[i, 1:feat_len+1] = torch.tensor(features[:feat_len]).float()
                                except Exception as feat_error:
                                    logger.debug(f"Error while processing atomic features: {feat_error}")
                                    continue
            
            try:
                edge_index = []
                threshold = 4.0 
                
                for i in range(num_atoms):
                    for j in range(num_atoms):
                        if i != j:  
                            dist = torch.norm(pos[i] - pos[j])
                            if dist < threshold:
                                edge_index.append([i, j])
                
                if edge_index:
                    edge_index = torch.tensor(edge_index).t().contiguous().long()
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
            except Exception as edge_error:
                logger.warning(f"Error while creating edge indices: {edge_error}; using an empty edge index.")
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            pyg_data = Data(
                pos=pos,          
                z=z,              
                x=x,             
                edge_index=edge_index  
            )
            
          
            if edge_index.size(1) > 0:
               
                FIXED_EDGE_FEATURE_DIM = 3  
                edge_attr = torch.zeros(edge_index.size(1), FIXED_EDGE_FEATURE_DIM)
                for i in range(edge_index.size(1)):
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    edge_attr[i, 0] = torch.norm(pos[src] - pos[dst])
                    edge_attr[i, 1] = 1.0
                pyg_data.edge_attr = edge_attr
            
            conformer_result['pyg_data'] = pyg_data
            
            conformer_result['node_feat'] = x.clone()
            
            
        return conformer_result
        
    except Exception as e:
        logger.warning(f"Failed to add a PyG object to conformer data: {e}")
        return conformer_result

def create_dataset_info_file(dataset_name, tasks, task_type):
    info_path = os.path.join(PROCESSED_DATA_DIR, dataset_name, 'dataset_info.json')
    
    info = {
        'dataset_name': dataset_name,
        'tasks': tasks,
        'task_type': task_type,
        'num_tasks': len(tasks)
    }
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Saved dataset information to {info_path} (task_type: {task_type})")

def load_and_prepare_dataset(dataset_name: str, data_dir: str) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
    processed_dir = os.path.join(data_dir, 'processed', dataset_name)
    logger.info(f"Attempting to load processed data from: {processed_dir}")

    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"Processed dataset directory does not exist: {processed_dir}. "
                                f"Run 'python data/prepare_datasets.py --dataset {dataset_name}' first to preprocess the dataset.")

    info_file = os.path.join(processed_dir, 'dataset_info.json')
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Dataset information file 'dataset_info.json' was not found in: {processed_dir}")
    try:
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        logger.info(f"Successfully loaded dataset information: {dataset_info}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing 'dataset_info.json': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading 'dataset_info.json': {e}")
        raise

    prepared_data = {'train': [], 'val': [], 'test': []}
    for split in ['train', 'val', 'test']:
        index_file = os.path.join(processed_dir, f'{split}_indices.json')
        if not os.path.exists(index_file):
            logger.warning(f"Index file '{split}_indices.json' was not found in: {processed_dir}. "
                           f"Skipping the '{split}' split.")
            prepared_data[split] = None
            continue

        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)

            num_samples = index_data.get('num_samples', 0)
            smiles_list = index_data.get('smiles', [])
            labels_list = index_data.get('labels', [])
            idx_mapping = index_data.get('idx_mapping', {})
            original_molecule_group_ids_from_json = index_data.get('original_molecule_group_ids', [])

            if not (num_samples == len(smiles_list) == len(labels_list) == len(idx_mapping) == len(original_molecule_group_ids_from_json)):
                logger.warning(f"Data lengths are inconsistent in '{split}_indices.json' "
                               f"(num_samples={num_samples}, smiles={len(smiles_list)}, "
                               f"labels={len(labels_list)}, idx_map={len(idx_mapping)}, "
                               f"group_ids={len(original_molecule_group_ids_from_json)}). This may cause errors.")

            split_samples = []
            for new_idx in range(num_samples):
                augmented_sample_id_val = idx_mapping.get(str(new_idx))
                if augmented_sample_id_val is None:
                    logger.warning(f"New index {new_idx} was not found in idx_mapping of '{split}_indices.json'; skipping this sample.")
                    continue

                if new_idx >= len(original_molecule_group_ids_from_json):
                    logger.warning(f"Index {new_idx} is out of range for original_molecule_group_ids in '{split}_indices.json'; skipping this sample.")
                    continue
                original_molecule_group_id_val = original_molecule_group_ids_from_json[new_idx]

                sample_labels_raw = labels_list[new_idx]
                if sample_labels_raw is None: sample_labels_raw = []
                processed_labels = [float(l) if l is not None else float('nan') for l in sample_labels_raw]

                sample_dict = {
                    'smiles': smiles_list[new_idx],
                    'labels': processed_labels,  
                    'augmented_sample_id': augmented_sample_id_val,
                    'original_molecule_id_for_files': original_molecule_group_id_val,
                    'original_molecule_group_id': original_molecule_group_id_val
                }
                split_samples.append(sample_dict)

            prepared_data[split] = split_samples
            logger.info(f"Successfully loaded {len(split_samples)} sample references for the '{split}' split.")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing '{split}_indices.json': {e}")
            prepared_data[split] = None
        except IndexError as e:
            logger.error(f"Index error while processing '{split}_indices.json' (possibly inconsistent data): {e}")
            prepared_data[split] = None
        except Exception as e:
            logger.error(f"Unexpected error while loading '{split}_indices.json': {e}")
            prepared_data[split] = None

    return prepared_data, dataset_info

def convert_pyg_to_graphormer_dict(pyg_data):
    """Convert PyG data to the dictionary format required by Graphormer."""
    try:
        if pyg_data is None:
            return {}
        
        x = pyg_data.x
        edge_index = pyg_data.edge_index
        edge_attr = getattr(pyg_data, 'edge_attr', None)
        in_degree = getattr(pyg_data, 'in_degree', None)
        out_degree = getattr(pyg_data, 'out_degree', None)
        spatial_pos = getattr(pyg_data, 'spatial_pos', None)
        
        FIXED_NODE_FEATURE_DIM = 9
        num_nodes = x.size(0)
        if x.size(1) != FIXED_NODE_FEATURE_DIM:
            new_x = torch.zeros((num_nodes, FIXED_NODE_FEATURE_DIM), dtype=torch.float, device=x.device)
            min_dim = min(x.size(1), FIXED_NODE_FEATURE_DIM)
            new_x[:, :min_dim] = x[:, :min_dim]
            x = new_x
        
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        
        FIXED_EDGE_FEATURE_DIM = 3
        if edge_attr is not None:
            if edge_attr.dim() != 2 or edge_attr.size(1) != FIXED_EDGE_FEATURE_DIM:
                num_edges = edge_attr.size(0)
                new_edge_attr = torch.zeros((num_edges, FIXED_EDGE_FEATURE_DIM), dtype=torch.float, device=edge_attr.device)
                if edge_attr.dim() == 2:
                    min_dim = min(edge_attr.size(1), FIXED_EDGE_FEATURE_DIM)
                    new_edge_attr[:, :min_dim] = edge_attr[:, :min_dim]
                else:
                    new_edge_attr[:, 0] = edge_attr
                edge_attr = new_edge_attr
        
        graphormer_dict = {
            'edge_index': edge_index,
            'node_features': x,
            'node_feat': x.clone(), 
            'x': x.clone()  
        }
        
        
        if edge_attr is not None:
            graphormer_dict['edge_attr'] = edge_attr
        else:
          
            graphormer_dict['edge_attr'] = torch.zeros((edge_index.size(1), FIXED_EDGE_FEATURE_DIM), dtype=torch.float, device=edge_index.device)
        
        if in_degree is not None:
            graphormer_dict['in_degree'] = in_degree
        else:
            
            graphormer_dict['in_degree'] = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
        
        if out_degree is not None:
            graphormer_dict['out_degree'] = out_degree
        else:
          
            graphormer_dict['out_degree'] = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        
        if spatial_pos is not None:
            graphormer_dict['spatial_pos'] = spatial_pos
        else:
        
            graphormer_dict['spatial_pos'] = torch.zeros((num_nodes, num_nodes), dtype=torch.long, device=x.device)
          
            for i in range(num_nodes):
                graphormer_dict['spatial_pos'][i, i] = 0
        
        return graphormer_dict
    except Exception as e:
        logger.error(f"Failed to convert PyG data to Graphormer format: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Molecular dataset preprocessing utility')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name to process, or "all" to process all supported datasets')
    parser.add_argument('--clean', action='store_true', help='Remove previous preprocessing outputs before processing')
    parser.add_argument('--config', type=str, default=None, help='Path to a configuration file')
    parser.add_argument('--dump-split-stats', action='store_true', help='Only compute and save train/val/test statistics without generating features')

    args = parser.parse_args()
    
    logger.info(f"Classification datasets: {CLASSIFICATION_DATASETS}")
    logger.info(f"Regression datasets: {REGRESSION_DATASETS}")

    tokenizer = None
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    global_max_seq_length = 128


    if args.dataset == 'all':
        process_all_datasets(tokenizer, clean=args.clean, default_max_seq_length=global_max_seq_length) 
    else:
        dataset_name_lower = args.dataset.lower()
        
        if dataset_name_lower in REGRESSION_DATASETS:
             task_type = 'regression'
        elif dataset_name_lower in CLASSIFICATION_DATASETS:
             task_type = 'classification'
        else:
             parser.error(
                 f"Unsupported dataset '{args.dataset}'. Supported datasets: "
                 f"{', '.join(sorted(SUPPORTED_DATASETS))}, or use 'all'."
             )

        logger.info(f"Inferred task type for dataset '{args.dataset}': {task_type}")

        enumeration_factor_to_use = 1
        scaffold_overlap_ratio_to_use = 0.0 
            
        config_file_path = None
        if args.config:
            possible_paths = [
                args.config,
                os.path.join(project_root_dir, 'configs', args.config),
                os.path.join(project_root_dir, args.config),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_file_path = path
                    logger.info(f"Using configuration file specified on the command line: {config_file_path}")
                    break
            if config_file_path is None:
                logger.warning(f"The configuration file specified on the command line was not found. Tried: {possible_paths}")
        
        if config_file_path is None:
            default_config_path = os.path.join(project_root_dir, 'configs', f"{args.dataset.lower()}.yaml")
            if os.path.exists(default_config_path):
                config_file_path = default_config_path
                logger.info(f"Using default configuration file: {config_file_path}")
            else:
                logger.warning(f"No configuration file found: command-line value {args.config if args.config else 'not specified'} or default path {default_config_path}")
        
        if config_file_path is not None and os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(
                    f"Loaded configuration file {config_file_path}; "
                    "the split method is fixed to scaffold (SMILES enumeration augmentation is disabled)."
                )
            except Exception as e_conf:
                logger.warning(
                    f"Unable to load or parse configuration file {config_file_path}: {e_conf}. "
                    "The split method remains fixed to scaffold."
                )
        else:
            logger.warning(
                f"Configuration file {config_file_path if config_file_path else 'not specified'} was not found. "
                "The split method remains fixed to scaffold."
            )

        if args.clean:
            clean_dataset_directories(args.dataset)

        current_max_seq_length = global_max_seq_length 
        specific_config_path = config_file_path
        if specific_config_path is not None and os.path.exists(specific_config_path):
            try:
                with open(specific_config_path, 'r') as f_spec_cfg:
                    spec_cfg = yaml.safe_load(f_spec_cfg)
                current_max_seq_length = spec_cfg.get('data',{}).get('max_seq_length', global_max_seq_length)

                logger.info(f"Loaded max_seq_length={current_max_seq_length} from configuration file: {specific_config_path}")
            except Exception as e_spec_cfg_max_len:
                logger.warning(f"Unable to load max_seq_length from {specific_config_path}: {e_spec_cfg_max_len}. Using: {current_max_seq_length}")
        else:
             logger.info(f"Configuration file {specific_config_path if specific_config_path else 'not specified'} was not found. Using max_seq_length: {current_max_seq_length}")

        process_dataset(
            args.dataset, 
            task_type, 
            tokenizer, 
            max_seq_length=current_max_seq_length,
            split_type='scaffold',
            scaffold_overlap_ratio=scaffold_overlap_ratio_to_use,
            dump_split_stats=args.dump_split_stats
        )

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to sys.path: {project_root}")
    
    if not logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized in __main__")

    main()
