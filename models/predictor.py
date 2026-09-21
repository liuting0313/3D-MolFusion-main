from typing import List, Optional

import torch
import torch.nn as nn


class Predictor(nn.Module):

    def __init__(
        self,
        embed_dim: int = 768,
        num_tasks: int = 1,
        task_type: str = "classification",
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if task_type not in {"classification", "regression"}:
            raise ValueError(
                "task_type must be either 'classification' or 'regression'."
            )

        self.task_type = task_type
        self.num_tasks = num_tasks

        layers = []
        input_dim = embed_dim
        for hidden_dim in hidden_dims or []:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_tasks))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
