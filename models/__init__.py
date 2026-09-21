import logging
from typing import Any, Dict

from .cross_modal_attention import CrossModalAttention, create_cross_modal_attention
from .direction_aware_autoencoder import (
    DirectionAwareConvAutoencoder,
    create_direction_aware_autoencoder,
)
from .fingerprint_mlp import FingerprintEncoder
from .molecular_positional_encoding import (
    GPSEncoderWithMolPE,
    GaussianRBF,
    MolecularPositionalEncoding,
    create_gps_encoder_with_mol_pe,
)
from .multiview_3dmol_model import MultiView3DMolModel, create_multiview_3dmol_model
from .multiview_renderer import MultiViewRendererTorch, create_multiview_renderer
from .predictor import Predictor

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_TYPES = {
    "multiview_3dmol",
    "multiview",
    "mv3d",
}


def create_model(
    config: Dict[str, Any], model_type: str = "multiview_3dmol"
) -> MultiView3DMolModel:
    """Create a 3D-MolFusion model from a model configuration dictionary."""
    model_type_lower = model_type.lower()
    if model_type_lower not in _SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported types: {supported}."
        )

    logger.info("Creating MultiView3DMolModel")
    return create_multiview_3dmol_model(config)


def get_model_info(model_type: str = "multiview_3dmol") -> Dict[str, Any]:
    """Return concise architecture metadata for logging and experiment records."""
    model_type_lower = model_type.lower()
    if model_type_lower not in _SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported types: {supported}."
        )

    return {
        "name": "MultiView3DMolModel",
        "description": (
            "Multimodal molecular property predictor combining six-view 3D "
            "visual features, a geometry-enhanced graph encoder, and molecular "
            "fingerprints."
        ),
        "features": [
            "Six orthogonal molecular views",
            "Direction-aware 3D convolutional encoding",
            "Local GNN and global Transformer graph encoding",
            "Distance, angle, and relative-direction geometric biases",
            "Cross-modal attention and adaptive modality gating",
        ],
    }


__all__ = [
    "CrossModalAttention",
    "DirectionAwareConvAutoencoder",
    "FingerprintEncoder",
    "GPSEncoderWithMolPE",
    "GaussianRBF",
    "MolecularPositionalEncoding",
    "MultiView3DMolModel",
    "MultiViewRendererTorch",
    "Predictor",
    "create_cross_modal_attention",
    "create_direction_aware_autoencoder",
    "create_gps_encoder_with_mol_pe",
    "create_model",
    "create_multiview_3dmol_model",
    "create_multiview_renderer",
    "get_model_info",
]
