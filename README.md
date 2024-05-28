# Implementation of KAN with full MLP
The MLPKAN is a modified implementation of the KAN architecture, replacing traditional components like B-Spline or FFT (https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py) with complete MLP. This change potentially accelerates the model compared to its FourierKAN counterpart.

## Architecture Overview

MLPKAN operates in two stages:
1. **Dimensionality Expansion**: Input dimension `d_in` is expanded to `d_in * hidden_dim`. Unlike standard approaches, MLPKAN transforms `d_in` into a shape of `(d_in, 1)`, leveraging a weight matrix of shape `(1, hidden_dim)` to reduce parameters.
2. **Activation and Mapping**: The output from the first stage is passed through an activation function. Subsequently, it is mapped from `(d_in * hidden_dim, d_out)`.

Total parameters for MLPKAN are calculated as `hidden_dim * (1 + d_in + d_out)`, compared to the typical MLP method (achieving the same dimension transformation): `hidden_dim * d_in * (d_in + d_out)`.

# Performance after 1 epoch with different activation functions (97% max) 

```
Testing with ReLU
Epoch 1/1 Running Loss: 0.016869: 100%|██████████| 938/938 [00:13<00:00, 67.29it/s]

Test set: Average loss: 0.0007, Accuracy: 9503/10000 (95%)

Testing with GELU
Epoch 1/1 Running Loss: 0.123740: 100%|██████████| 938/938 [00:12<00:00, 73.98it/s]

Test set: Average loss: 0.0005, Accuracy: 9586/10000 (96%)

Testing with LeakyReLU
Epoch 1/1 Running Loss: 0.067448: 100%|██████████| 938/938 [00:13<00:00, 68.66it/s]

Test set: Average loss: 0.0005, Accuracy: 9633/10000 (96%)

Testing with Sigmoid
Epoch 1/1 Running Loss: 1.352230: 100%|██████████| 938/938 [00:13<00:00, 68.13it/s]

Test set: Average loss: 0.0060, Accuracy: 3658/10000 (37%)

Testing with Tanh
Epoch 1/1 Running Loss: 0.039728: 100%|██████████| 938/938 [00:13<00:00, 67.79it/s]

Test set: Average loss: 0.0007, Accuracy: 9509/10000 (95%)

Testing with ELU
Epoch 1/1 Running Loss: 0.145392: 100%|██████████| 938/938 [00:13<00:00, 69.59it/s]

Test set: Average loss: 0.0005, Accuracy: 9651/10000 (97%)

Testing with Cos
Epoch 1/1 Running Loss: 0.223601: 100%|██████████| 938/938 [00:14<00:00, 65.44it/s]

Test set: Average loss: 0.0011, Accuracy: 9154/10000 (92%)

Testing with SiLU
Epoch 1/1 Running Loss: 0.120909: 100%|██████████| 938/938 [00:14<00:00, 65.96it/s]

Test set: Average loss: 0.0005, Accuracy: 9566/10000 (96%)
```
The model achieves the best performance with the ELU activation function, reaching an accuracy of 97%.
