# 3D-MolFusion

[![Python](https://img.shields.io/badge/Python-3.8.20-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

3D-MolFusion is a multimodal framework for molecular property prediction. It combines a six-view 3D visual representation, a geometry-enhanced 2D molecular graph, and molecular fingerprints through cross-modal attention and adaptive fusion.

## Supported benchmarks

The repository contains dataset-specific configurations for eight MoleculeNet benchmarks:

| Task | Datasets |
|---|---|
| Classification | BACE, BBBP, ClinTox, SIDER, Tox21 |
| Regression | ESOL, FreeSolv, Lipophilicity |

All configurations use deterministic Bemis–Murcko scaffold splitting.

## Repository structure

```text
3D-MolFusion/
├── configs/                 # Dataset-specific experiment configurations
├── data/                    # Preprocessing, conformer generation and data loading
├── models/                  # Model components and the 3D-MolFusion architecture
├── scripts/
│   ├── train.py             # Training entry point
│   └── test.py              # Evaluation entry point
├── utils/                   # Losses and evaluation metrics
├── requirements.txt         # Minimal pinned runtime dependencies
└── LICENSE                  # MIT license
```

## Environment

The reference experiments were conducted with:

- Ubuntu 20.04, x86-64
- Python 3.8
- PyTorch 2.4.1 built with CUDA 11.8
- PyTorch Geometric 2.6.1
- RDKit 2022.09.5
- NumPy 1.24.4
- NVIDIA RTX A6000 GPU

Install the runtime dependencies in a clean environment:

```bash
conda create -n 3dmolfusion python=3.8.20 -y
conda activate 3dmolfusion
pip install -r requirements.txt
pip check
```

The PyTorch Geometric extension wheels must match PyTorch 2.4 and CUDA 11.8. If they are not resolved automatically, install them from the matching wheel index:

```bash
pip install torch-scatter==2.1.2+pt24cu118 \
  torch-sparse==0.6.18+pt24cu118 \
  torch-cluster==1.6.3+pt24cu118 \
  torch-spline-conv==1.2.2+pt24cu118 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

## Data preprocessing

The default data root is `./dataset`. Prepare one benchmark with:

```bash
python data/prepare_datasets.py --dataset BACE --clean
```

Prepare all eight benchmarks with:

```bash
python data/prepare_datasets.py --dataset all --clean
```

Processed files are stored under:

```text
dataset/processed/<DATASET>/
```

## Training

Train a benchmark from the project root:

```bash
python scripts/train.py --config configs/bace.yaml
```

Other examples:

```bash
python scripts/train.py --config configs/tox21.yaml
python scripts/train.py --config configs/esol.yaml
```

An optional seed override is available:

```bash
python scripts/train.py --config configs/bace.yaml --seed_override 666
```

Each run creates a timestamped experiment directory containing:

```text
experiments/<EXPERIMENT_NAME>_<DATASET>_<TIMESTAMP>/
├── config.yaml
├── checkpoints/
│   ├── model_best.pth.tar
│   └── checkpoint_last.pth.tar
└── training_summary.json
```

`config.yaml` is copied verbatim into the run directory. Checkpoints contain the model state, optimizer state, scheduler state, dataset metadata and the monitored best validation metric.

## Evaluation

Evaluate a saved checkpoint with:

```bash
python scripts/test.py \
  --experiment_dir experiments/<RUN_DIRECTORY> \
  --checkpoint_name model_best.pth.tar \
  --split test
```

Reported metrics are:

- Classification: ROC-AUC, accuracy and F1 score
- Regression: RMSE and MAE on the original target scale

## Reproducibility

Dataset configurations specify the training seed, batch size, epoch budget, optimizer, scheduler, dropout and early-stopping policy. The training script seeds Python, NumPy, PyTorch and CUDA and configures deterministic cuDNN behavior.


Because GPU kernels and library builds can still introduce small numerical differences, report the exact configuration, package environment and hardware together with experimental results.

## License

This project is released under the [MIT License](LICENSE).
