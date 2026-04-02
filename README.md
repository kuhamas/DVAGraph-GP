# DVAGraph-GP
# DVAGraph-GP: Improving phenotypic prediction by multi-scale genomic compression and dynamic graph learning

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Overview
DVAGraph‑GP is a deep learning framework for genomic prediction that simultaneously addresses two fundamental challenges:

1 Multi‑scale genomic compression – preserving local‑to‑long‑range SNP patterns while reducing dimensionality.

2 Complex interaction modeling – learning trait‑specific SNP interaction graphs directly from the data.

 # The model combines three modules:

1- Dilated Convolutional Fusion Variational Autoencoder (DCF‑VAE) – compresses high‑dimensional SNP data using parallel dilated convolutions (dilation rates 2, 4, 6) without increasing parameters, and provides probabilistic regularization via KL divergence.

2- Dynamic Adaptive Graph Spectral Convolution (DAGSC) – builds a dynamic graph over the latent features using an ELU‑softmax adjacency matrix, then applies Chebyshev spectral convolution (order 2) to capture local and global interactions.

3- Global Pooling & Regression – aggregates graph‑enhanced features via global mean pooling and maps them to phenotype predictions through a fully connected layer.

** The whole model is trained end‑to‑end with a composite loss: prediction (MSE) + reconstruction (MSE) + KL divergence.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Key Innovations
*Multi‑scale dilated convolutions – capture local (d=2), intermediate (d=4) and long‑range (d=6) genomic patterns without increasing parameter count.

*Dynamic adaptive graph learning – the graph adjacency matrix is learned from the latent features and updated during training, adapting to trait‑specific interactions.

*Chebyshev spectral convolution – approximates spectral graph convolution without expensive eigendecomposition, enabling efficient propagation of information up to 2‑hop neighborhoods.

*Unified hyperparameters – the same hyperparameter set works across all traits and datasets, eliminating trait‑specific tuning.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Requirements
torch==2.4.1
torch-geometric==2.6.1
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
scipy==1.10.1
matplotlib==3.7.2
seaborn==0.13.2
