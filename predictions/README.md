# Saved test predictions

`test_predictions.npz` of each of the 18 models, in the same layout as the training output:

```
{architecture}/{scheme}/seed{n}/test_predictions.npz
```

- architecture: `resnet50`, `efficientnet_b0`
- scheme: `4class` (flat model), `3class` (stage 1), `2class` (stage 2)
- seed: 0, 1, 2

Each file contains `prob` (softmax probabilities, 1,600 × number of classes), `y_true` (four-class labels) and `paths` (test image paths, used to check the image order).
