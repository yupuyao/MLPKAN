# Implementation of KAN with full MLP
The MLPKAN is a modified implementation of the KAN architecture, replacing traditional components like B-Spline or FFT (https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py) with complete MLP. This change potentially accelerates the model compared to its FourierKAN counterpart.

## Architecture Overview

MLPKAN operates in three stages:
1. **Dimensionality Expansion**: Input dimension `d_in` is expanded to `d_in * hidden_dim`. Unlike standard approaches, MLPKAN transforms `d_in` into a shape of `(d_in, 1)`, leveraging a weight matrix of shape `(1, hidden_dim)` to reduce parameters.
2. **Activation Learning**: The output from the first stage is passed through an activation function. Subsequently, it is mapped from `(hidden_dim, 1)`.
3. **Feature Mapping**: Apply the usual MLP map feature dimension from `d_in` into `d_out`.

Total parameters for MLPKAN are calculated as `2 * hidden_dim + d_in * d_out`, compared to the typical MLP method (achieving the same dimension transformation): `hidden_dim * d_in * (d_in + d_out)`.

# Performance after 1 epoch with different activation functions (96% max) 

```
Testing with ReLU
Epoch 1/1 Running Loss: 0.246414: 100%|██████████| 938/938 [00:14<00:00, 63.20it/s]
Test set: Average loss: 0.0005, Accuracy: 9580/10000 (96%)

Testing with GELU
Epoch 1/1 Running Loss: 0.285726: 100%|██████████| 938/938 [00:13<00:00, 69.55it/s]
Test set: Average loss: 0.0005, Accuracy: 9588/10000 (96%)

Testing with LeakyReLU
Epoch 1/1 Running Loss: 0.166177: 100%|██████████| 938/938 [00:13<00:00, 69.83it/s]
Test set: Average loss: 0.0005, Accuracy: 9582/10000 (96%)

Testing with Sigmoid
Epoch 1/1 Running Loss: 0.267998: 100%|██████████| 938/938 [00:13<00:00, 67.21it/s]
Test set: Average loss: 0.0008, Accuracy: 9417/10000 (94%)

Testing with Tanh
Epoch 1/1 Running Loss: 0.066968: 100%|██████████| 938/938 [00:13<00:00, 70.41it/s]
Test set: Average loss: 0.0006, Accuracy: 9538/10000 (95%)

Testing with ELU
Epoch 1/1 Running Loss: 0.122461: 100%|██████████| 938/938 [00:15<00:00, 60.85it/s]
Test set: Average loss: 0.0005, Accuracy: 9589/10000 (96%)

Testing with Cos
Epoch 1/1 Running Loss: 0.103279: 100%|██████████| 938/938 [00:14<00:00, 65.47it/s]

Test set: Average loss: 0.0005, Accuracy: 9639/10000 (96%)

Testing with SiLU
Epoch 1/1 Running Loss: 0.262126: 100%|██████████| 938/938 [00:13<00:00, 67.50it/s]

Test set: Average loss: 0.0007, Accuracy: 9480/10000 (95%)

```
