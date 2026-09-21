from collections import defaultdict
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = True) -> str:
    """Generate a Bemis-Murcko scaffold for one SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality,
    )


def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """Split molecules into disjoint scaffold groups deterministically."""
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    n_samples = len(smiles_list)
    scaffold_to_indices = defaultdict(list)

    for idx, smiles in enumerate(smiles_list):
        scaffold_to_indices[generate_scaffold(smiles, include_chirality=True)].append(idx)

    scaffold_sets = [
        sorted(indices)
        for _, indices in sorted(
            scaffold_to_indices.items(),
            key=lambda item: (len(item[1]), item[1][0]),
            reverse=True,
        )
    ]

    train_cutoff = frac_train * n_samples
    valid_cutoff = (frac_train + frac_valid) * n_samples
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert not set(train_idx).intersection(valid_idx)
    assert not set(train_idx).intersection(test_idx)
    assert not set(valid_idx).intersection(test_idx)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == n_samples
    return train_idx, valid_idx, test_idx


def split_dataframe(dataframe, smiles_column: str = "smiles"):
    """Split a DataFrame using the deterministic scaffold protocol."""
    train_idx, valid_idx, test_idx = scaffold_split(dataframe[smiles_column].tolist())
    return (
        dataframe.iloc[train_idx].reset_index(drop=True),
        dataframe.iloc[valid_idx].reset_index(drop=True),
        dataframe.iloc[test_idx].reset_index(drop=True),
    )


def get_scaffold_set(dataframe, smiles_column: str = "smiles"):
    """Return the set of scaffolds present in a DataFrame."""
    return {
        generate_scaffold(smiles, include_chirality=True)
        for smiles in dataframe[smiles_column]
    }


def verify_split(train_df, valid_df, test_df, smiles_column: str = "smiles"):
    """Assert that no scaffold is shared between the three splits."""
    train_scaffolds = get_scaffold_set(train_df, smiles_column)
    valid_scaffolds = get_scaffold_set(valid_df, smiles_column)
    test_scaffolds = get_scaffold_set(test_df, smiles_column)
    assert train_scaffolds.isdisjoint(valid_scaffolds)
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert valid_scaffolds.isdisjoint(test_scaffolds)
    print("Scaffold leakage check: PASSED")
    print(f"Train scaffolds: {len(train_scaffolds)}")
    print(f"Valid scaffolds: {len(valid_scaffolds)}")
    print(f"Test scaffolds: {len(test_scaffolds)}")


__all__ = [
    "generate_scaffold",
    "scaffold_split",
    "split_dataframe",
    "get_scaffold_set",
    "verify_split",
]
