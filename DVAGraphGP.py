# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing import load_data, convert_to_tensor
import pandas as pd

# VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # Encoder layers
        self.enc_fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm_fc1 = nn.LayerNorm(2048)

        # Dilated convolutional paths with proper LayerNorm
        self.dilation_path1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.LayerNorm(1024)  # Applied to spatial dimension
        )

        self.dilation_path2 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.LayerNorm(1024)  # Applied to spatial dimension
        )

        self.dilation_path3 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, padding=6, dilation=6),
            nn.GELU(),
            nn.LayerNorm(1024)  # Applied to spatial dimension
        )

        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(64 + 128 + 256, 256, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(1024)  # Applied to spatial dimension
        )

        # Dynamically calculate conv output size
        with torch.no_grad():
            dummy = torch.randn(2, input_dim)
            dummy = self.enc_fc1(dummy)
            dummy = F.gelu(dummy)
            dummy = self.layer_norm_fc1(dummy)
            dummy = dummy.view(2, 1, -1)
            dummy = F.max_pool1d(dummy, kernel_size=2, stride=2)

            d1 = self.dilation_path1(dummy)
            d2 = self.dilation_path2(dummy)
            d3 = self.dilation_path3(dummy)

            fused = torch.cat([d1, d2, d3], dim=1)
            fused = self.feature_fusion(fused)

            self.conv_output_size = fused.view(2, -1).size(1)

        # Latent projection layers
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        # Decoder layers
        self.dec_fc1 = nn.Linear(latent_dim, 1024)
        self.dec_fc2 = nn.Linear(1024, 2048)
        self.dec_fc3 = nn.Linear(2048, input_dim)

    def encode(self, x):
        h = F.gelu(self.enc_fc1(x))
        h = self.layer_norm_fc1(h)

        # Reshape and reduce dimension
        h = h.view(-1, 1, 2048)
        h = F.max_pool1d(h, kernel_size=2, stride=2)

        # Process through parallel dilation paths
        d1 = self.dilation_path1(h)
        d2 = self.dilation_path2(h)
        d3 = self.dilation_path3(h)

        # Concatenate features along channel dimension
        fused = torch.cat([d1, d2, d3], dim=1)
        h = self.feature_fusion(fused)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.gelu(self.dec_fc1(z))
        h = F.gelu(self.dec_fc2(h))
        return self.dec_fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z


# Adaptive Graph Learning
class AdaptiveGraphLearning(nn.Module):
    def forward(self, z):
        B, K = z.size()
        z = z.unsqueeze(1)
        sim_matrix = torch.bmm(z.transpose(1, 2), z)
        sim_matrix = F.elu(sim_matrix)
        return F.softmax(sim_matrix, dim=-1)


# Graph Convolution Network (GCN)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, cheb_K=3):
        super(GraphConvolution, self).__init__()
        self.cheb_K = cheb_K
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z, adj):
        B, K, in_features = z.size()
        D = torch.sum(adj, dim=-1, keepdim=True)
        D_inv_sqrt = 1.0 / torch.sqrt(D + 1e-8)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        I = torch.eye(K, device=z.device).unsqueeze(0).repeat(B, 1, 1)
        L = I - adj

        # Correct Laplacian normalization
        L_tilde = D_inv_sqrt * L
        L_tilde = L_tilde * D_inv_sqrt.permute(0, 2, 1)

        # Chebyshev polynomials
        Tx_0 = z
        Tx_1 = torch.bmm(L_tilde, z)
        cheb_terms = [Tx_0, Tx_1]
        for k in range(2, self.cheb_K):
            Tx_k = 2 * torch.bmm(L_tilde, cheb_terms[-1]) - cheb_terms[-2]
            cheb_terms.append(Tx_k)

        total = sum(cheb_terms)
        out = self.fc(total)
        out = self.dropout(out)
        return out


# Global Pooling Fully Connected Layer
class GlobalPoolingFC(nn.Module):
    def __init__(self, embedding_dim):
        super(GlobalPoolingFC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, X_g):
        X_pooled = X_g.mean(dim=1)  # Global mean pooling
        return self.fc(X_pooled)


# DVAGraph-GP Model Definition
class DVAGraphGP(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(DVAGraphGP, self).__init__()
        self.vae = VAE(input_dim, latent_dim)
        self.graph = AdaptiveGraphLearning()
        self.gcn = GraphConvolution(in_features=1, out_features=64)
        self.global_pooling_fc = GlobalPoolingFC(64)

    def forward(self, x):
        z = self.vae(x)
        A = self.graph(z)
        z_gcn = z.unsqueeze(-1)
        gcn_out = self.gcn(z_gcn, A)
        return self.global_pooling_fc(gcn_out)
