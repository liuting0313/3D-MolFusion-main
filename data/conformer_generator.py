import logging
from typing import Any, Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

MMFF_MAX_ITERS = 200
CONFORMER_MAX_ITERS = MMFF_MAX_ITERS

CONFORMER_GENERATION_STATS = {
    "total_attempts": 0,
    "initial_etkdg_success": 0,
    "fallback_etkdg_success": 0,
    "mmff_converged": 0,
    "mmff_non_converged": 0,
    "mmff_unavailable": 0,
    "missing_conformer": 0,
}


def print_conformer_generation_stats() -> Dict[str, int]:
    """Log and return a snapshot; reporting must not reset accumulated counts."""
    stats = dict(CONFORMER_GENERATION_STATS)
    logger.info("Conformer generation statistics: %s", stats)
    return stats


def _etkdg_parameters(seed: int):
    initial = AllChem.ETKDGv3()
    initial.randomSeed = int(seed)
    initial.maxAttempts = 1000

    random_coords = AllChem.ETKDGv3()
    random_coords.randomSeed = int(seed + 1)
    random_coords.useRandomCoords = True
    random_coords.maxAttempts = 2000

    extended = AllChem.ETKDGv3()
    extended.randomSeed = int(seed + 2)
    extended.maxAttempts = 5000

    return (initial, random_coords, extended)


def generate_single_conformer(
    smiles: str,
    seed: int = 42,
    *,
    include_mol: bool = False,
) -> Optional[Dict[str, Any]]:
    CONFORMER_GENERATION_STATS["total_attempts"] += 1

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        CONFORMER_GENERATION_STATS["missing_conformer"] += 1
        return None

    mol = Chem.AddHs(mol)
    successful_attempt = None
    for attempt_index, params in enumerate(_etkdg_parameters(seed)):
        mol.RemoveAllConformers()
        try:
            status = AllChem.EmbedMolecule(mol, params)
        except (RuntimeError, ValueError):
            status = -1
        if status == 0:
            successful_attempt = attempt_index
            break

    if successful_attempt is None:
        CONFORMER_GENERATION_STATS["missing_conformer"] += 1
        return None

    if successful_attempt == 0:
        CONFORMER_GENERATION_STATS["initial_etkdg_success"] += 1
    else:
        CONFORMER_GENERATION_STATS["fallback_etkdg_success"] += 1

    mmff_status = "unavailable"
    try:
        properties = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant="MMFF94s"
        )
        if properties is not None:
            optimization_status = AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94s",
                maxIters=MMFF_MAX_ITERS,
            )
            if optimization_status == 0:
                mmff_status = "converged"
            else:
                mmff_status = "non-converged"
    except (RuntimeError, ValueError):
        mmff_status = "unavailable"

    CONFORMER_GENERATION_STATS[f"mmff_{mmff_status.replace('-', '_')}"] += 1

    full_coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    full_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    retained_mol = Chem.RemoveHs(mol)
    conformer = retained_mol.GetConformer()
    coordinates = np.asarray(conformer.GetPositions(), dtype=np.float32)
    atomic_numbers = [atom.GetAtomicNum() for atom in retained_mol.GetAtoms()]

    result: Dict[str, Any] = {
        "success": True,
        "conformers": [{
            "coordinates": coordinates,
            "atoms": [retained_mol.GetAtomWithIdx(i).GetSymbol() for i in range(retained_mol.GetNumAtoms())],
            "coordinates_with_h": full_coordinates,
            "atoms_with_h": full_atoms,
            "atomic_numbers": atomic_numbers,
            "num_atoms": len(atomic_numbers),
            "generation_method": "ETKDGv3",
            "embedding_attempt": "initial" if successful_attempt == 0 else "fallback",
            "optimization_method": "MMFF94s",
            "mmff_status": mmff_status,
            "mmff_max_iterations": MMFF_MAX_ITERS,
        }],
        "mmff_status": mmff_status,
    }
    if include_mol:
        result["_mol"] = retained_mol
        result["_mol_with_h"] = mol
    return result


class ConformerGenerator:

    def __init__(self, **kwargs):
        self.seed = int(kwargs.get("seed", 42))
        requested_method = str(kwargs.get("optimization_method", "MMFF")).upper()
        if requested_method not in {"MMFF", "MMFF94S"}:
            raise ValueError("Only the specified MMFF94s optimization is supported.")
        if kwargs.get("enable_database", False):
            logger.warning("Using the ETKDGv3 fallback.")

    def generate_conformers(
        self, smiles: str, mol_id: Optional[str] = None
    ) -> Dict[str, Any]:
        seed = self.seed
        if mol_id is not None:
            try:
                seed += int(mol_id)
            except (TypeError, ValueError):
                pass
        result = generate_single_conformer(smiles, seed=seed)
        if result is None:
            return {
                "success": False,
                "conformers": [],
                "error": "All ETKDGv3 embedding attempts failed",
            }
        return result


__all__ = [
    "ConformerGenerator",
    "CONFORMER_GENERATION_STATS",
    "MMFF_MAX_ITERS",
    "CONFORMER_MAX_ITERS",
    "generate_single_conformer",
    "print_conformer_generation_stats",
]
