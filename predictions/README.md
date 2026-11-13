# Saved predictions

Saved predictions of every model, in the same layout as the training output. Trained weights are not included.

## main/

Flat, stage-1, and stage-2 models of the main experiment:

```
main/{architecture}/{scheme}/seed{n}/test_predictions.npz
```

- scheme: `4class` (flat model), `3class` (stage 1), `2class` (stage 2)

## exploratory/

Models of the exploratory experiments:

```
exploratory/{experiment}/{architecture}/{scheme}/seed{n}/
    test_predictions.npz
    val_predictions.npz
    history.csv
```

| Experiment | Scheme | Models |
| ---------- | ------ | ------ |
| `class_weighted` | `3class_weighted` (stage 1 trained with class-weighted cross-entropy) | 6 |
| `control_pair` | `3class_ctrl` (stage 1 with peaberry and longberry merged), `2class_ctrl` (longberry vs peaberry) | 12 |
| `no_color_jitter` | `4class`, `3class`, `2class` (trained without color jitter) | 18 |

## Common to all folders

- architecture: `resnet50`, `efficientnet_b0`
- seed: 0, 1, 2
- Each `.npz` file contains `prob` (softmax probabilities, number of images × number of classes), `y_true` (four-class labels), and `paths` (image paths, used to check the image order). Test files cover the 1,600 test images and validation files the 1,600 validation images.
- `history.csv` lists the training and validation loss and accuracy of each epoch.
