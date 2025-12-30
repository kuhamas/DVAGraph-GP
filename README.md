# DVAGraph-GP
DVAGraph-GP: Improving Phenotypic Prediction by Non-linear Dimension Reduction and Complex Genomic Relationship Modeling
# Overview
DVAGraph-GP (Dilated Convolution Variational Autoencoder Graph for Genomic Prediction) is a deep learning framework designed to enhance genomic prediction in plant breeding. It leverages a combination of dilated convolutions, variational autoencoders (VAE), and graph neural networks (GNN) to capture complex, non-linear relationships between genotype and phenotype data. This model is optimized for genomic selection tasks, particularly for datasets with a large number of features and limited sample sizes, overcoming challenges such as the "curse of dimensionality."

The model uses a consistent set across all traits, simplifying training and improving efficiency and scalability. Finally, our experiments were executed using the PyTorch library (version 2.4.1) on an Anaconda environment within a Windows 11 Home workstation. The workstation is burdened with an Intel Xeon Gold 6133 CPU, 32 GB RAM, and an NVIDIA GeForce RTX 4060 GPU with 8 GB VRAM, accelerated using CUDA 12.4. Python 3.8.20 was used, along with the required libraries of torchvision, torchaudio, numpy, pandas, and scikit-learn to train and test deep learning models successfully

# Requirements
torch==2.4.1
torch-geometric==2.6.1
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
scipy==1.10.1
matplotlib==3.7.2
seaborn==0.13.2
