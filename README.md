# ANN Backpropagation Implementation
 
A from-scratch implementation of a feedforward neural network with backpropagation, written in both pure Python and NumPy. Developed for **22118 Advanced Python and Unix for Bioinformaticians** at the Technical University of Denmark (DTU).
 
## Overview
 
This project implements a multilayer perceptron (MLP) trained via backpropagation, applied to a binary classification problem from DTU HealthTech: predicting whether single nucleotide polymorphism (SNP) variants are disease-associated.
 
Two implementations are provided to compare a from-first-principles approach against a vectorised one:
 
- **Pure Python** (`src/base_python_version/`) — uses only the standard library, with weights and activations represented as nested lists. Intended to make every step of forward and backward propagation explicit.
- **NumPy** (`src/numpy_version/`) — the same algorithm, refactored to use NumPy arrays and matrix operations.
Each layer includes a bias neuron, and the network supports configurable architectures, activation functions, loss functions, mini-batch gradient descent, L2 regularisation, learning rate decay, and early stopping based on validation loss.
 
## Repository Structure
 
```
ANN_Backprop_Implement/
├── data/                       # Training and test datasets
├── models/                     # Saved weights and biases
├── results/                    # Entry-point scripts
│   ├── main_base_python.py
│   └── main_numpy.py
├── src/
│   ├── base_python_version/    # Pure Python implementation
│   └── numpy_version/          # NumPy implementation
├── test/                       # Unit tests (pytest)
└── README.md
```
 
## Requirements
 
- Python 3.10+
- NumPy (for the NumPy implementation only)
- pytest (for running tests)
Install dependencies with:
 
```bash
pip install -r requirements.txt
```
 
## Usage
 
Check our report!
 
## Implementation Notes
 
- The ANN architecture (number of layers, neurons per layer, activation functions) is configured at instantiation.
- Training uses mini-batch stochastic gradient descent with optional L2 regularisation and exponential learning rate decay.
- Class imbalance in the training set is addressed via minority class oversampling; the validation and test sets are left untouched.
- Early stopping monitors validation loss and restores the best parameters when patience is exceeded.
- The two implementations use different random number generators (`random` for pure Python, `numpy.random` for NumPy), so identical seeds do not produce identical results across implementations.
## References
 
- DTU HealthTech, *Artificial Neural Network* lecture notes: [https://teaching.healthtech.dtu.dk/22118/index.php/Artificial_Neural_Network](https://teaching.healthtech.dtu.dk/22118/index.php/Artificial_Neural_Network)
## Authors
 
- André Godinho — `s253707`
- Lun Suša — `s253705`
## Course
 
22118 Advanced Python and Unix for Bioinformaticians — DTU, Spring 2026
