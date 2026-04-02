import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Load wheat dataset (TKW)
# ---------------------------


# ---------------------------
# Hyperparameters
# ---------------------------
latent_dim = 64
batch_size = 32
vae_epochs = 50
gcn_epochs = 200
learning_rate_vae = 1e-4
learning_rate_gcn = 1e-3
weight_decay = 1e-5
n_folds = 10
random_state = 42
trait_name = 'GP'
top_k = 30                     # keep top 30 strongest GRM edges per node

# ---------------------------
# VAE (same as in your paper)
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.enc_fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm_fc1 = nn.LayerNorm(2048)

        self.dilation_path1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.LayerNorm(1024)
        )
        self.dilation_path2 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.LayerNorm(1024)
        )
        self.dilation_path3 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, padding=6, dilation=6),
            nn.GELU(),
            nn.LayerNorm(1024)
        )
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(64+128+256, 256, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(1024)
        )

        with torch.no_grad():
            dummy = torch.randn(2, input_dim)
            dummy = F.gelu(self.enc_fc1(dummy))
            dummy = self.layer_norm_fc1(dummy)
            dummy = dummy.view(2, 1, -1)
            dummy = F.max_pool1d(dummy, kernel_size=2, stride=2)
            d1 = self.dilation_path1(dummy)
            d2 = self.dilation_path2(dummy)
            d3 = self.dilation_path3(dummy)
            fused = torch.cat([d1, d2, d3], dim=1)
            fused = self.feature_fusion(fused)
            self.conv_output_size = fused.view(2, -1).size(1)

        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, 1024)
        self.dec_fc2 = nn.Linear(1024, 2048)
        self.dec_fc3 = nn.Linear(2048, input_dim)

    def encode(self, x):
        h = F.gelu(self.enc_fc1(x))
        h = self.layer_norm_fc1(h)
        h = h.view(-1, 1, 2048)
        h = F.max_pool1d(h, kernel_size=2, stride=2)
        d1 = self.dilation_path1(h)
        d2 = self.dilation_path2(h)
        d3 = self.dilation_path3(h)
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
        return z, mu, logvar

# ---------------------------
# Simple GCN (Kipf & Welling style)
# ---------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden_dim=64, out_dim=1):
        super(SimpleGCN, self).__init__()
        self.gc1 = nn.Linear(in_features, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        # adj: normalized adjacency (N, N)
        # x: (N, in_features)
        x = torch.mm(adj, x)          # first propagation
        x = F.relu(self.gc1(x))
        x = torch.mm(adj, x)          # second propagation
        x = self.gc2(x)
        return x.squeeze(-1)

# ---------------------------
# Build GRM adjacency (sparse top-k)
# ---------------------------
def build_grm_adj(X_all, p_j, k=30):
    """
    X_all: combined genotypes (train+val) shape (N, m)
    p_j: allele frequencies from training set (length m)
    Returns normalized adjacency matrix as torch tensor.
    """
    # Center genotypes
    Z = X_all - 2 * p_j
    denom = 2 * np.sum(p_j * (1 - p_j))
    if denom == 0:
        denom = 1e-8
    G = (Z @ Z.T) / denom
    np.fill_diagonal(G, 0)               # remove self-loops initially
    # Keep top k absolute values per row (symmetric)
    n = G.shape[0]
    adj = np.zeros_like(G)
    for i in range(n):
        row = G[i]
        abs_row = np.abs(row)
        idx = np.argsort(abs_row)[-k:]   # indices of top k
        adj[i, idx] = row[idx]
    # Make symmetric (average)
    adj = (adj + adj.T) / 2
    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    # Symmetric normalization
    D = np.sum(adj, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-8)
    D_inv_sqrt = np.diag(D_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return torch.tensor(adj_norm, dtype=torch.float32)

# ---------------------------
# Output directory
# ---------------------------
#output_dir = r''
os.makedirs(output_dir, exist_ok=True)

kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold+1} ===")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # ---------- Train VAE ----------
    print("  Training VAE...")
    vae = VAE(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate_vae, weight_decay=weight_decay)
    vae_dataset = TensorDataset(X_train_t)
    vae_loader = DataLoader(vae_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(vae_epochs):
        vae.train()
        total_loss = 0
        for (batch_x,) in vae_loader:
            batch_x = batch_x.to(device)
            vae_optimizer.zero_grad()
            z, mu, logvar = vae(batch_x)
            recon = vae.decode(z)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            vae_optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    VAE epoch {epoch+1}: loss = {total_loss/len(vae_loader):.4f}")

    # ---------- Extract latent means ----------
    vae.eval()
    with torch.no_grad():
        train_mu = vae.encode(X_train_t.to(device))[0].cpu().numpy()
        val_mu = vae.encode(X_val_t.to(device))[0].cpu().numpy()

    # ---------- Build GRM adjacency ----------
    X_all = np.vstack([X_train, X_val])
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    # Frequencies from training set only
    p_j = np.mean(X_train, axis=0) / 2.0
    p_j = np.clip(p_j, 1e-6, 1 - 1e-6)
    adj_norm = build_grm_adj(X_all, p_j, k=top_k).to(device)

    # ---------- Prepare node features ----------
    all_mu = np.vstack([train_mu, val_mu])
    X_feat = torch.tensor(all_mu, dtype=torch.float32).to(device)

    # ---------- Train GCN ----------
    print("  Training GCN on GRM graph...")
    gcn = SimpleGCN(in_features=latent_dim).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=learning_rate_gcn, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    for epoch in range(gcn_epochs):
        gcn.train()
        optimizer.zero_grad()
        pred_all = gcn(X_feat, adj_norm)
        pred_train = pred_all[:n_train]
        loss = criterion(pred_train, y_train_t)
        if torch.isnan(loss):
            print(f"    NaN loss at epoch {epoch+1}. Stopping training for this fold.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gcn.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch+1) % 40 == 0:
            print(f"    GCN epoch {epoch+1}: loss = {loss.item():.4f}")

    # ---------- Evaluate ----------
    gcn.eval()
    with torch.no_grad():
        pred_all = gcn(X_feat, adj_norm)
        pred_val = pred_all[n_train:].cpu().numpy()

    if np.any(np.isnan(pred_val)):
        pred_val = np.full_like(y_val, np.mean(y_train))

    mse = mean_squared_error(y_val, pred_val)
    corr, _ = pearsonr(y_val, pred_val)
    if np.isnan(corr):
        corr = 0.0
    fold_results.append([mse, corr])
    print(f"    Fold {fold+1} results: MSE = {mse:.4f}, Corr = {corr:.4f}")

# ---------------------------
# Save results
# ---------------------------
fold_results = np.array(fold_results)
avg_mse = np.mean(fold_results[:, 0])
std_mse = np.std(fold_results[:, 0])
avg_corr = np.mean(fold_results[:, 1])
std_corr = np.std(fold_results[:, 1])

print("\n=== Final Results (GRM Graph, GP) ===")
print(f"MSE: {avg_mse:.4f} ± {std_mse:.4f}")
print(f"Corr: {avg_corr:.4f} ± {std_corr:.4f}")

df_folds = pd.DataFrame(fold_results, columns=['MSE', 'Corr'])
df_folds.to_csv(os.path.join(output_dir, f'{trait_name}_fold_results.csv'), index=False)

with open(os.path.join(output_dir, f'{trait_name}_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"GRM Graph (VanRaden, 2008, top-{top_k}) on wheat {trait_name}\n")
    f.write(f"10-fold CV\n")
    f.write(f"MSE: {avg_mse:.4f} ± {std_mse:.4f}\n")
    f.write(f"Corr: {avg_corr:.4f} ± {std_corr:.4f}\n")

# ---------------------------
# Visualize GRM adjacency (first fold)
# ---------------------------
print("\nGenerating GRM adjacency heatmap for first fold...")
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
train_idx, _ = next(kf.split(X))
X_train_fold = X[train_idx]
p_j_fold = np.mean(X_train_fold, axis=0) / 2.0
p_j_fold = np.clip(p_j_fold, 1e-6, 1 - 1e-6)
adj_viz = build_grm_adj(X_train_fold, p_j_fold, k=top_k).numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(adj_viz, cmap='viridis', cbar=True, square=True,
            xticklabels=False, yticklabels=False)
plt.title(f'GRM Graph (top-{top_k} edges) – first fold')
plt.tight_layout()
heatmap_path = os.path.join(output_dir, f'grm_graph_heatmap_{trait_name}.pdf')
plt.savefig(heatmap_path, dpi=500, format='pdf', bbox_inches='tight')
plt.savefig(heatmap_path.replace('.pdf', '.eps'), dpi=500, format='eps', bbox_inches='tight')
plt.show()
print(f"GRM graph heatmap saved to {heatmap_path} and .eps")