# Two-Stage Coffee Classification

Two-Stage Hierarchical Convolutional Neural Network for Green Arabica Coffee Bean Image Classification

## Overview

Comparison of flat and two-stage hierarchical classification of green arabica coffee bean images:

- **Flat**: one CNN with four outputs (defect, longberry, peaberry, premium)
- **Two-stage**: Stage 1 (defect, longberry, peaberry + premium) → Stage 2 (peaberry vs premium)
- **Models**: ResNet-50 and EfficientNet-B0, three seeds each (18 models)
- **Evaluation**: accuracy, per-class metrics, McNemar test

## Data

- **Dataset**: USK-Coffee, https://coffee.comvislab-usk.org/ (not included)
- **Split**: fixed split of the dataset (1,200 / 400 / 400 images per class)

## Training Settings

| Setting      | Value                                                        |
| ------------ | ------------------------------------------------------------ |
| Input        | 224 × 224                                                    |
| Augmentation | horizontal flip, random resized crop, color jitter, rotation |
| Optimizer    | Adam, learning rate 1e-4                                     |
| Batch size   | 32                                                           |
| Epochs       | max 30, early stopping (patience 5)                          |
| Seeds        | 0, 1, 2                                                      |

## Usage

```
notebooks/01_training_and_validation.ipynb   # Train the 18 models
notebooks/02_test_evaluation.ipynb           # Test evaluation and McNemar test
```

To reproduce the test results without training, set `OUT_ROOT` in notebook 02 to `predictions/`.

## Structure

```
├── notebooks/       # Training and evaluation notebooks
├── predictions/     # Saved test predictions
├── results/         # Training histories and test results
└── requirements.txt
```

## Requirements

- Python 3.13+
- PyTorch 2.11+
- CUDA (optional, for GPU training)

## Usage Permission

Free to use, modify, and share.