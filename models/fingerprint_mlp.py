from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerprintEncoder(nn.Module):
    MACCS_DIM = 167
    PUBCHEM_DIM = 881
    MORGAN_DIM = 2048
    SOURCE_NAMES = ("maccs", "pubchem", "morgan")

    def __init__(
        self,
        input_dim: int = 3096,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        patch_size: int = 24,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("FingerprintEncoder requires num_layers >= 1.")
        if patch_size < 1:
            raise ValueError("FingerprintEncoder requires patch_size >= 1.")

        expected_dim = self.MACCS_DIM + self.PUBCHEM_DIM + self.MORGAN_DIM
        if input_dim != expected_dim:
            raise ValueError(
                f"Segmented fingerprint tokenization requires input_dim={expected_dim}; "
                f"received {input_dim}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.segment_specs: List[Tuple[str, int]] = [
            ("maccs", self.MACCS_DIM),
            ("pubchem", self.PUBCHEM_DIM),
            ("morgan", self.MORGAN_DIM),
        ]

        self.segment_offsets = {}
        start = 0
        for name, segment_dim in self.segment_specs:
            self.segment_offsets[name] = (start, start + segment_dim)
            start += segment_dim

        self.segment_num_patches = {
            name: (segment_dim + self.patch_size - 1) // self.patch_size
            for name, segment_dim in self.segment_specs
        }
        self.total_num_patches = sum(self.segment_num_patches.values())

        layers = [
            nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout_rate),
                ]
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.segment_patch_projs = nn.ModuleDict(
            {
                name: nn.Linear(self.patch_size, output_dim)
                for name, _ in self.segment_specs
            }
        )
        self.segment_position_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.randn(1, self.segment_num_patches[name], output_dim)
                    * 0.02
                )
                for name, _ in self.segment_specs
            }
        )

        self.source_embedding = nn.Embedding(len(self.segment_specs), output_dim)
        self.source_to_idx = {
            name: index for index, (name, _) in enumerate(self.segment_specs)
        }
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.token_norm = nn.LayerNorm(output_dim)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unsupported activation: {activation}")

    def _segment_to_patch_tokens(
        self,
        segment_tensor: torch.Tensor,
        segment_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, segment_dim = segment_tensor.shape
        num_patches = self.segment_num_patches[segment_name]
        padded_dim = num_patches * self.patch_size

        if segment_dim < padded_dim:
            segment_tensor = F.pad(
                segment_tensor,
                (0, padded_dim - segment_dim),
                mode="constant",
                value=0.0,
            )

        patches = segment_tensor.view(batch_size, num_patches, self.patch_size)
        patch_tokens = self.segment_patch_projs[segment_name](patches)
        patch_tokens = (
            patch_tokens
            + self.segment_position_embeddings[segment_name][:, :num_patches, :]
        )

        source_ids = torch.full(
            (batch_size, num_patches),
            fill_value=self.source_to_idx[segment_name],
            dtype=torch.long,
            device=segment_tensor.device,
        )
        patch_tokens = patch_tokens + self.source_embedding(source_ids)

        patch_starts = (
            torch.arange(num_patches, device=segment_tensor.device)
            * self.patch_size
        )
        patch_mask = (patch_starts < segment_dim).unsqueeze(0).expand(batch_size, -1)
        return patch_tokens, patch_mask

    def _build_patch_tokens(
        self, fingerprints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_segments = []
        mask_segments = []
        for segment_name, _ in self.segment_specs:
            start, end = self.segment_offsets[segment_name]
            tokens, mask = self._segment_to_patch_tokens(
                fingerprints[:, start:end], segment_name
            )
            token_segments.append(tokens)
            mask_segments.append(mask)

        patch_tokens = torch.cat(token_segments, dim=1)
        patch_mask = torch.cat(mask_segments, dim=1)
        patch_tokens = self.token_norm(
            patch_tokens + self.token_encoder(patch_tokens)
        )
        return patch_tokens, patch_mask

    def forward(
        self,
        fingerprints: torch.Tensor,
        *,
        return_sequence: bool = False,
        return_both: bool = False,
    ) -> torch.Tensor:
        if fingerprints.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected fingerprint dimension {self.input_dim}; "
                f"received {fingerprints.shape[-1]}."
            )

        patch_tokens, patch_mask = self._build_patch_tokens(fingerprints)
        mask_f = patch_mask.unsqueeze(-1).to(patch_tokens.dtype)
        pooled_tokens = (patch_tokens * mask_f).sum(dim=1) / mask_f.sum(
            dim=1
        ).clamp(min=1.0)

        mlp_feature = self.layer_norm(self.mlp(fingerprints))
        graph_feature = self.layer_norm(0.5 * (mlp_feature + pooled_tokens))

        if return_both:
            return graph_feature, patch_tokens, patch_mask
        if return_sequence:
            return patch_tokens
        return graph_feature
