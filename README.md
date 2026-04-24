# Self-Pruning Neural Network for CIFAR-10

This repository contains the implementation of a feed-forward neural network that learns to prune itself dynamically during training via **learnable sigmoid gates and L1 regularization**.

## Overview

The core of this project is the custom `PrunableLinear` layer. In this layer, each weight $w_{ij}$ is multiplied by a learnable gate, $\sigma(g_{ij})$. An L1 penalty is added to the training loss to encourage the gate values $g_{ij}$ to be pushed towards large negative values, effectively bringing the multiplicative gates $\sigma(g_{ij})$ to 0. Unimportant connections thus get dynamically pruned over the course of training, making the model more lightweight while retaining high accuracy.

The model is trained on the CIFAR-10 dataset to demonstrate the trade-off between sparsity (percentage of pruned connections) and test accuracy, controlled by a hyperparameter $\lambda$.

## Features

- **Custom `PrunableLinear` Layer**: Includes learnable gates that apply continuous native pruning.
- **Dynamic Sparse Regularization**: The overall loss incorporates the L1 norm of the gate activations: `loss = cross_entropy + lambda * sum(gates)`.
- **Sparsity-Accuracy Trade-off Tracking**: Capabilities to track, visualize, and compare model sparsity against predictive correctness across different $\lambda$ penalties.

## Repository Contents

- `notebooks/self_pruning_nn_cifar10.ipynb`: Jupyter/Colab Notebook containing the complete end-to-end pipeline: setup, model class, data loaders, training utilities, and evaluation.
- `REPORT.md`: A detailed report on the empirical results, model statistics, and experimental trade-offs.
- `results/results_summary.json`: JSON summary of metrics over various hyperparameter settings.
- `assets/*.png`: Matplotlib plots tracking model metrics across epochs and histogram visuals of connection gate distributions.

## Setup and Usage

### Requirements
- Python 3.8+
- `torch`, `torchvision` (PyTorch ecosystem)
- `matplotlib`, `numpy`, `jupyter`

### Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd self-pruning-nn
   ```
2. Install the necessary dependencies (via pip):
   ```bash
   pip install torch torchvision matplotlib numpy jupyter
   ```

### Running the Project

You can run the full pipeline using the provided Jupyter Notebook:
1. Open the `notebooks/self_pruning_nn_cifar10.ipynb` notebook via Jupyter Lab or import it directly into Google Colab.
2. Run all cells to process the dataset, evaluate permutations, and render new visualization charts directly in the UI.

## Acknowledgements 

This notebook was originally parameterized for compute execution on an NVIDIA T4 GPU within Google Colab environments.
