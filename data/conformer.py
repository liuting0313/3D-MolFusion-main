from __future__ import absolute_import, division, print_function

import os
import logging
import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool

try:
    from numba import njit
except ImportError:
    # Optional accelerator; the function below also works as ordinary Python.
    def njit(function):
        return function
from tqdm import tqdm

logger = logging.getLogger(__name__)


allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


class ConformerGen(object):

    def __init__(self, **params):
        self._init_features(**params)

    def _init_features(self, **params):
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        # Preserve the pretrained vocabulary: never invent replacement token IDs.
        self.dictionary = params.get('dictionary')
        if self.dictionary is None:
            try:
                from unimol_tools.data.conformer import ConformerGen as UniMolConformerGen
            except ImportError as exc:
                raise ImportError(
                    "This optional Uni-Mol adapter requires dictionary=<matching "
                    "Uni-Mol dictionary> or the unimol-tools package. The main "
                    "3D-MolFusion pipeline uses data.conformer_generator instead."
                ) from exc
            self.dictionary = UniMolConformerGen(**params).dictionary
        self.dictionary.add_symbol("[MASK]", is_special=True)
        if os.name == 'posix':
            self.multi_process = params.get('multi_process', True)
        else:
            self.multi_process = params.get('multi_process', False)
            if self.multi_process:
                logger.warning(
                    'Please use "if __name__ == "__main__":" to wrap the main function when using multi_process on Windows.'
                )

    def single_process(self, smiles):
        if self.method == 'rdkit_random':
            atoms, coordinates, mol = inner_smi2coords(
                smiles, seed=self.seed, mode=self.mode, remove_hs=self.remove_hs
            )
            feat = coords2unimol(
                atoms,
                coordinates,
                self.dictionary,
                self.max_atoms,
                remove_hs=self.remove_hs,
            )
            return feat, mol
        else:
            raise ValueError(
                'Unknown conformer generation method: {}'.format(self.method)
            )

    def transform_raw(self, atoms_list, coordinates_list):

        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform_mols(self, mols_list):
        inputs = []
        for mol in mols_list:
            atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform(self, smiles_list):
        if len(smiles_list) == 0:
            return [], []
        logger.info('Start generating conformers...')
        if self.multi_process:
            pool = Pool(processes=min(8, os.cpu_count()))
            results = [
                item for item in tqdm(pool.imap(self.single_process, smiles_list))
            ]
            pool.close()
        else:
            results = [self.single_process(smiles) for smiles in tqdm(smiles_list)]

        inputs, mols = zip(*results)
        inputs = list(inputs)
        mols = list(mols)

        failed_conf = [(item['src_coord'] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf)) * 100
            )
        )
        failed_conf_indices = [
            index for index, value in enumerate(failed_conf) if value
        ]
        if len(failed_conf_indices) > 0:
            logger.info('Failed conformers indices: {}'.format(failed_conf_indices))
            logger.debug(
                'Failed conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_indices]
                )
            )

        failed_conf_3d = [(item['src_coord'][:, 2] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf_3d)) * 100
            )
        )
        failed_conf_3d_indices = [
            index for index, value in enumerate(failed_conf_3d) if value
        ]
        if len(failed_conf_3d_indices) > 0:
            logger.info(
                'Failed 3d conformers indices: {}'.format(failed_conf_3d_indices)
            )
            logger.debug(
                'Failed 3d conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_3d_indices]
                )
            )
        return inputs, mols


def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True, return_mol=False):
    from data.conformer_generator import generate_single_conformer

    result = generate_single_conformer(smi, seed=seed, include_mol=True)
    if result is None:
        raise RuntimeError(f'All ETKDGv3 embedding attempts failed: {smi}')
    conformer_entry = result['conformers'][0]
    atoms = list(conformer_entry['atoms_with_h'])
    coordinates = np.asarray(
        conformer_entry['coordinates_with_h'], dtype=np.float32
    )
    assert len(atoms) > 0, 'No atoms in molecule: {}'.format(smi)

    # Retain the original bonds and the conformer in the returned molecule.
    mol = result['_mol'] if remove_hs else result['_mol_with_h']
    if return_mol:
        return mol

    assert len(atoms) == len(
        coordinates
    ), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h, mol
    else:
        return atoms, coordinates, mol


def inner_coords(atoms, coordinates, remove_hs=True):
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates


def coords2unimol(
    atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params
):
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    src_tokens = np.array(
        [dictionary.bos()]
        + [dictionary.index(atom) for atom in atoms]
        + [dictionary.eos()]
    )
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1, 3)), src_coord, np.zeros((1, 3))], axis=0)
    src_distance = distance_matrix(src_coord, src_coord)
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(
        1, -1
    )

    return {
        'src_tokens': src_tokens.astype(int),
        'src_distance': src_distance.astype(np.float32),
        'src_coord': src_coord.astype(np.float32),
        'src_edge_type': src_edge_type.astype(int),
    }


class UniMolV2Feature(object):
    def __init__(self, **params):
        self._init_features(**params)

    def _init_features(self, **params):
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 128)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', True)
        if os.name == 'posix':
            self.multi_process = params.get('multi_process', True)
        else:
            self.multi_process = params.get('multi_process', False)
            if self.multi_process:
                logger.warning(
                    'Please use "if __name__ == "__main__":" to wrap the main function when using multi_process on Windows.'
                )

    def single_process(self, smiles):
        if self.method == 'rdkit_random':
            mol = inner_smi2coords(
                smiles,
                seed=self.seed,
                mode=self.mode,
                remove_hs=self.remove_hs,
                return_mol=True,
            )
            feat = mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs)
            return feat, mol
        else:
            raise ValueError(
                'Unknown conformer generation method: {}'.format(self.method)
            )

    def transform_raw(self, atoms_list, coordinates_list):

        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            mol = create_mol_from_atoms_and_coords(atoms, coordinates)
            inputs.append(mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform_mols(self, mols_list):
        inputs = []
        for mol in mols_list:
            inputs.append(mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform(self, smiles_list):
        if len(smiles_list) == 0:
            return [], []
        logger.info('Start generating conformers...')
        if self.multi_process:
            pool = Pool(processes=min(8, os.cpu_count()))
            results = [
                item for item in tqdm(pool.imap(self.single_process, smiles_list))
            ]
            pool.close()
        else:
            results = [self.single_process(smiles) for smiles in tqdm(smiles_list)]

        inputs, mols = zip(*results)
        inputs = list(inputs)
        mols = list(mols)

        failed_conf = [(item['src_coord'] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf)) * 100
            )
        )
        failed_conf_indices = [
            index for index, value in enumerate(failed_conf) if value
        ]
        if len(failed_conf_indices) > 0:
            logger.info('Failed conformers indices: {}'.format(failed_conf_indices))
            logger.debug(
                'Failed conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_indices]
                )
            )

        failed_conf_3d = [(item['src_coord'][:, 2] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf_3d)) * 100
            )
        )
        failed_conf_3d_indices = [
            index for index, value in enumerate(failed_conf_3d) if value
        ]
        if len(failed_conf_3d_indices) > 0:
            logger.info(
                'Failed 3d conformers indices: {}'.format(failed_conf_3d_indices)
            )
            logger.debug(
                'Failed 3d conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_3d_indices]
                )
            )

        return inputs, mols


def create_mol_from_atoms_and_coords(atoms, coordinates):
    mol = Chem.RWMol()
    atom_indices = []

    for atom in atoms:
        atom_idx = mol.AddAtom(Chem.Atom(atom))
        atom_indices.append(atom_idx)

    conf = Chem.Conformer(len(atoms))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord)

    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    return mol


def mol2unimolv2(mol, max_atoms=128, remove_hs=True, **params):

    mol = Chem.RemoveHs(mol) if remove_hs else Chem.Mol(mol)
    atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)

    if len(atoms) > max_atoms:
        mask = np.zeros(len(atoms), dtype=bool)
        mask[:max_atoms] = True
        np.random.shuffle(mask) 
        atoms = atoms[mask]
        coordinates = coordinates[mask]
    else:
        mask = np.ones(len(atoms), dtype=bool)
    src_tokens = [AllChem.GetPeriodicTable().GetAtomicNumber(item) for item in atoms]
    src_coord = coordinates
    node_attr, edge_index, edge_attr = get_graph(mol)
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat=0, mask=mask)
    feat['src_tokens'] = src_tokens
    feat['src_coord'] = src_coord
    return feat


def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol):
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    num_bond_features = 3  
    if len(mol.GetBonds()) > 0:  
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = np.array(edges_list, dtype=np.int32).T
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr


def get_graph_features(edge_attr, edge_index, node_attr, drop_feat, mask):
    atom_feat_sizes = [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    edge_attr, edge_index, x = edge_attr, edge_index, node_attr
    N = x.shape[0]

    atom_feat = convert_to_single_emb(x[:, 1:], atom_feat_sizes)

    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr, edge_feat_sizes) + 1
    )
    shortest_path_result = floyd_warshall(adj)
    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        shortest_path_result[...] = 511
    else:
        atom_feat = atom_feat + 2
        edge_feat = edge_feat + 2
        degree = degree + 2
        shortest_path_result = shortest_path_result + 1

    feat = {}
    feat["atom_feat"] = atom_feat[mask]
    feat["atom_mask"] = np.ones(N, dtype=np.int64)[mask]
    feat["edge_feat"] = edge_feat[mask][:, mask]
    feat["shortest_path"] = shortest_path_result[mask][:, mask]
    feat["degree"] = degree.reshape(-1)[mask]
    atoms = atom_feat[..., 0]
    pair_type = np.concatenate(
        [
            np.expand_dims(atoms, axis=(1, 2)).repeat(N, axis=1),
            np.expand_dims(atoms, axis=(0, 2)).repeat(N, axis=0),
        ],
        axis=-1,
    )
    pair_type = pair_type[mask][:, mask]
    feat["pair_type"] = convert_to_single_emb(pair_type, [128, 128])
    feat["attn_bias"] = np.zeros((mask.sum() + 1, mask.sum() + 1), dtype=np.float32)
    return feat


def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


@njit
def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 510

    for i in range(n):
        M[i, i] = 0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj

    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                M[i, j] = 510
    return M
