# Two-Stage Coffee Classification

Two-Stage Hierarchical Convolutional Neural Network for Green Arabica Coffee Bean Image Classification

## Overview

Comparison of flat and two-stage hierarchical classification of green arabica coffee bean images:

- **Flat**: one CNN with four outputs (defect, longberry, peaberry, premium)
- **Two-stage**: Stage 1 (defect, longberry, peaberry + premium) → Stage 2 (peaberry vs premium)
- **Models**: ResNet-50 and EfficientNet-B0; for each architecture and seed (0, 1, 2), a flat, a stage-1, and a stage-2 model are trained, giving 18 models in total
- **Evaluation**: accuracy, per-class metrics, exact McNemar test with Holm correction, 95% confidence intervals across seeds

The main experiment (notebooks 01–02) is complemented by a data audit (03), exploratory experiments added after the main evaluation (04–07), and a measurement of the computational cost (08).

## Data

- **Dataset**: USK-Coffee, https://coffee.comvislab-usk.org/ (not included)
- **Split**: fixed split of the dataset (1,200 / 400 / 400 images per class)

## Training Settings

| Setting          | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| Input            | 224 × 224, ImageNet normalization                            |
| Augmentation     | horizontal flip, random resized crop, color jitter, rotation |
| Optimizer        | Adam, learning rate 1e-4, halved every 5 epochs              |
| Batch size       | 32                                                           |
| Epochs           | max 30, early stopping (patience 5)                          |
| Weight selection | epoch with the lowest validation loss                        |
| Seeds            | 0, 1, 2                                                      |

## Notebooks

| Notebook | Content | Results |
| -------- | ------- | ------- |
| `01_training_and_validation.ipynb` | Train the main flat, stage-1, and stage-2 models | `results/main/training/` |
| `02_test_evaluation.ipynb` | Test evaluation, McNemar test, confusion matrices | `results/main/test/` |
| `03_data_audit.ipynb` | Split check and duplicate search across train, val, and test | `results/audit/` |
| `04_stage1_class_weighted.ipynb` | Stage 1 retrained with class-weighted cross-entropy (exploratory) | `results/exploratory/class_weighted/` |
| `05_val_test_gap.ipynb` | Analysis of the validation–test gap (exploratory) | `results/exploratory/val_test_gap/` |
| `06_control_pair.ipynb` | Control version merging peaberry with longberry (exploratory) | `results/exploratory/control_pair/` |
| `07_no_color_jitter.ipynb` | All models retrained without color jitter (exploratory) | `results/exploratory/no_color_jitter/` |
| `08_inference_overhead.ipynb` | Inference time, model size, and training time of the main models | `results/main/inference_overhead/` |

## Reproducing Without Training

Trained weights are not included. To reproduce the results from the saved predictions, set `OUT_ROOT` to `predictions/main/` (and, in notebooks 04, 06, and 07, the folder of the new models to the matching folder in `predictions/exploratory/`), then skip the training sections. The dataset is still needed.

## Structure

```
├── notebooks/            # Notebooks 01–08
├── predictions/
│   ├── main/             # Test predictions of the main models
│   └── exploratory/      # Predictions and training histories of the class-weighted, control-pair, and no-color-jitter models
├── results/
│   ├── main/             # Training histories, test results, and inference overhead of the main models
│   ├── exploratory/      # Class weighting, control pair, no color jitter, and validation–test gap
│   └── audit/            # Split check and duplicate search
└── requirements.txt
```

## Requirements

- Python 3.13+
- PyTorch 2.11+
- CUDA (optional, for GPU training)

## Usage Permission

Free to use, modify, and share.
