# A Hybrid Framework Using Transformer and Physics-Informed Neural Operator for Correcting Systematic Biases in TEMPO NO₂ Columns across North America

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code accompanying the manuscript for training and evaluating a hybrid machine learning framework to correct systematic biases in TEMPO satellite NO₂ column retrievals using Pandora ground-based measurements.

## Overview

The proposed framework combines:
- **Transformer encoder** for processing surface and viewing geometry features
- **Fourier Neural Operator (FNO)** for learning vertical profile representations
- **Cross-attention fusion** to integrate spectral and spatial information

## Installation

```bash
git clone https://github.com/username/NO2_Bias_Correction.git
cd NO2_Bias_Correction
pip install -r requirements.txt
```

### Dependencies
- PyTorch ≥ 2.0
- xarray
- pandas
- numpy
- scikit-learn
- neuralop
- ray *(optional, for distributed training)*

## Data

Training requires collocated TEMPO-Pandora observations. Place the following in `data/`:
- `combined_ds.nc` — Preprocessed collocated dataset

## Usage

### Training

```bash
# K-fold cross-validation
python K_fold_with_dates.py

# Leave-one-station-out evaluation
python LOSO_eval.py
```

### Evaluation

```bash
python sample_eval.py
```

## Repository Structure

```
├── models/              # Model architectures
├── utils/               # Training utilities
├── eval_notebooks/      # Analysis notebooks
├── data/                # Data directory (not tracked)
├── Train.py             # Single training run
├── K_fold_with_dates.py # K-fold cross-validation
├── LOSO_eval.py         # Leave-one-station-out evaluation
└── sample_eval.py       # Evaluation script
```


## License

This project is licensed under the MIT License.
