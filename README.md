# Implementation of KAN with full MLP
The MLPKAN is a modified implementation of the KAN architecture, replacing components like B-Spline or FFT with complete MLP. This change potentially accelerates the model compared to its FourierKAN counterpart.

## Architecture Overview

MLPKAN Layer operates in three stages:
1. **Basis Generation**: The input with dimensions `(b, d_in, 1)` is first mapped to a higher-dimensional space using a weight matrix of dimensions `(1, hidden_dim)`. This mapping results in feature representations of dimensions `(b, d_in, hidden_dim)`.These high-dimensional features are then processed through a nonlinear function, transforming them into a set of bases within the space defined by the activation function we want to learn.
2. **Activation Function Learning**: The features serving as bases undergo further linear combination using a new linear transformation matrix of dimensions `(hidden_dim, 1)`. This transformation produces a single output for each input, resulting in a learnable activation function. The output after this step has dimensions `(b, d_in)`. (Instead of pure MLP, we could replace it with kernel learning, by mapping the input into two tensors with the same shape `(b, d_in, hidden_dim)`, and the inner product of the dimension `hidden_dim` could generate a kernel.)
3. **Feature Mapping**:  the output feature from the activation function learning stage, with dimensions `(b, d_in)`, is mapped to the final output dimensions `(b, d_out)`.

Total parameters for MLPKAN Layer are calculated as `2 * hidden_dim + d_in * d_out`.

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
