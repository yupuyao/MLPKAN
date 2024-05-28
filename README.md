# Implementation of KAN with full MLP
The MLPKAN is a modified implementation of the KAN architecture, replacing traditional components like B-Spline or FFT (https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py) with complete MLP. This change potentially accelerates the model compared to its FourierKAN counterpart.

## Architecture Overview

MLPKAN operates in three stages:
1. **Dimensionality Expansion**: Input dimension `d_in` is expanded to `d_in * hidden_dim`. Unlike standard approaches, MLPKAN transforms `d_in` into a shape of `(d_in, 1)`, leveraging a weight matrix of shape `(1, hidden_dim)` to reduce parameters.
2. **Activation Function Learning**: The output from the first stage is passed through an activation function. Subsequently, it is mapped from `(hidden_dim, 1)`.
3. **Feature Mapping**: Apply the normal MLP mapping feature dimension from `d_in` into `d_out`.

Total parameters for MLPKAN are calculated as `2 * hidden_dim + d_in * d_out`, compared to the typical MLP method (achieving the same dimension transformation): `hidden_dim * d_in * (d_in + d_out)`.

# Performance after 1 epoch with different activation functions (97% max) 

```
Epoch 1/1 Running Loss: 0.024769: 100%|██████████| 938/938 [00:16<00:00, 57.68it/s]
Test set: Average loss: 0.0005, Accuracy: 9586/10000 (96%)

Testing with GELU
Epoch 1/1 Running Loss: 0.464900: 100%|██████████| 938/938 [00:15<00:00, 62.32it/s]

Test set: Average loss: 0.0006, Accuracy: 9533/10000 (95%)

Testing with LeakyReLU
Epoch 1/1 Running Loss: 0.076915: 100%|██████████| 938/938 [00:15<00:00, 61.86it/s]

Test set: Average loss: 0.0005, Accuracy: 9608/10000 (96%)

Testing with Sigmoid
Epoch 1/1 Running Loss: 0.420319: 100%|██████████| 938/938 [00:14<00:00, 63.30it/s]

Test set: Average loss: 0.0008, Accuracy: 9440/10000 (94%)

Testing with Tanh
Epoch 1/1 Running Loss: 0.348992: 100%|██████████| 938/938 [00:14<00:00, 65.54it/s]
Test set: Average loss: 0.0006, Accuracy: 9493/10000 (95%)

Testing with ELU
Epoch 1/1 Running Loss: 0.142619: 100%|██████████| 938/938 [00:14<00:00, 64.46it/s]
Test set: Average loss: 0.0005, Accuracy: 9625/10000 (96%)

Testing with Cos
Epoch 1/1 Running Loss: 0.073093: 100%|██████████| 938/938 [00:15<00:00, 60.37it/s]

Test set: Average loss: 0.0004, Accuracy: 9658/10000 (97%)

Testing with SiLU
Epoch 1/1 Running Loss: 0.369693: 100%|██████████| 938/938 [00:13<00:00, 68.51it/s]

Test set: Average loss: 0.0006, Accuracy: 9545/10000 (95%)

```
