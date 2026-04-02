import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# -------------------- Reproducibility --------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Load Genotype Data --------------------
X = pd.read_csv('yourpath').values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# -------------------- Define Traits and Corresponding Files --------------------
traits = ['tkw', 'testw', 'length', 'width', 'Hard', 'Prot']
trait_to_file = {
    'tkw': 1,
    'testw': 2,
    'length': 3,
    'width': 4,
    'Hard': 5,
    'Prot': 6
}
base_pheno_path = 'yourpath_{}_phe.csv'

# -------------------- Variant Configuration --------------------
variant_name = 'Two_scale'
dilations = [2, 4]
kernel_size = 3

# -------------------- Safe Pearson Correlation --------------------
def safe_pearsonr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    try:
        return pearsonr(x, y)[0]
    except Exception:
        return 0.0

# -------------------- Define DCF_VAE (configurable) --------------------
class DCF_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dilations=[2], kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.dilations = dilations
        self.kernel_size = kernel_size

        self.enc_fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm_fc1 = nn.LayerNorm(2048)

        self.dilation_paths = nn.ModuleList()
        for d in dilations:
            padding = d * (kernel_size - 1) // 2
            self.dilation_paths.append(nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=kernel_size, padding=padding, dilation=d),
                nn.GELU(),
                nn.LayerNorm(1024)
            ))

        self.feature_fusion = nn.Sequential(
            nn.Conv1d(64 * len(dilations), 256, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm(1024)
        )

        with torch.no_grad():
            dummy = torch.randn(2, input_dim)
            dummy = F.gelu(self.enc_fc1(dummy))
            dummy = self.layer_norm_fc1(dummy)
            dummy = dummy.view(2, 1, -1)
            dummy = F.max_pool1d(dummy, kernel_size=2, stride=2)

            path_outputs = [path(dummy) for path in self.dilation_paths]
            fused = self.feature_fusion(torch.cat(path_outputs, dim=1))
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

        path_outputs = [path(h) for path in self.dilation_paths]
        fused = self.feature_fusion(torch.cat(path_outputs, dim=1))
        fused = fused.view(fused.size(0), -1)
        return self.fc_mu(fused), self.fc_logvar(fused)

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

# -------------------- Other Modules (unchanged) --------------------
class AdaptiveGraphLearning(nn.Module):
    def forward(self, z):
        B, K = z.size()
        z = z.unsqueeze(1)
        sim_matrix = torch.bmm(z.transpose(1, 2), z)
        sim_matrix = F.elu(sim_matrix)
        return F.softmax(sim_matrix, dim=-1)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, cheb_K=3):
        super().__init__()
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
        L_tilde = D_inv_sqrt * L
        L_tilde = L_tilde * D_inv_sqrt.permute(0, 2, 1)

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

class GlobalPoolingFC(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    def forward(self, X_g):
        X_pooled = X_g.mean(dim=1)
        return self.fc(X_pooled)

class SNPModel(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.graph = AdaptiveGraphLearning()
        self.gcn = GraphConvolution(in_features=1, out_features=64)
        self.global_pooling_fc = GlobalPoolingFC(64)

    def forward(self, x):
        z = self.vae(x)
        A = self.graph(z)
        z_gcn = z.unsqueeze(-1)
        gcn_out = self.gcn(z_gcn, A)
        return self.global_pooling_fc(gcn_out)

# -------------------- Output Directory --------------------
base_dir = r'yourpath'
variant_dir = os.path.join(base_dir, variant_name)
os.makedirs(variant_dir, exist_ok=True)

# -------------------- Loop Over All Traits --------------------
kf_outer = KFold(n_splits=10, shuffle=True, random_state=42)

for trait in traits:
    print(f"\n========== Processing trait: {trait} ==========")

    # Load phenotype data from the corresponding file
    file_num = trait_to_file[trait]
    pheno_path = base_pheno_path.format(file_num)

    try:
        df = pd.read_csv(pheno_path)
        y = df.iloc[:, 0].astype(float).values
    except Exception:
        df = pd.read_csv(pheno_path, header=None)
        y = df.iloc[:, 0].astype(float).values

    if np.any(np.isnan(y)):
        print(f"Warning: NaNs found in phenotype for {trait}. Removing them.")
        y = y[~np.isnan(y)]

    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(X_tensor)):
        print(f"  Fold {fold+1}/10")
        X_train = X_tensor[train_idx]
        y_train = y_tensor[train_idx]
        X_val = X_tensor[val_idx]
        y_val = y_tensor[val_idx]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

        vae = DCF_VAE(input_dim=X.shape[1], latent_dim=64,
                      dilations=dilations, kernel_size=kernel_size)
        model = SNPModel(vae).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        start_time = time.time()
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                z = model.vae(batch_X)
                recon = model.vae.decode(z)
                mu, logvar = model.vae.encode(batch_X)

                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                A = model.graph(z)
                gcn_out = model.gcn(z.unsqueeze(-1), A)
                y_pred = model.global_pooling_fc(gcn_out)

                recon_loss = F.mse_loss(recon, batch_X)
                pred_loss = F.mse_loss(y_pred, batch_y)
                total_loss = pred_loss + 0.1 * recon_loss + 0.01 * kl_div

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)

        epoch_time = (time.time() - start_time) / 100

        model.eval()
        preds, truths, recon_errors = [], [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                z = model.vae(batch_X)
                recon = model.vae.decode(z)
                recon_errors.append(F.mse_loss(recon, batch_X).item())
                y_out = model(batch_X)
                preds.extend(y_out.cpu().numpy().flatten())
                truths.extend(batch_y.cpu().numpy().flatten())

        preds = np.array(preds)
        truths = np.array(truths)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)) or np.any(np.isnan(truths)) or np.any(np.isinf(truths)):
            print(f"Warning: NaNs or infs detected in predictions/truths for fold {fold+1}. Setting correlation to 0.")
            corr = 0.0
        else:
            corr = safe_pearsonr(truths, preds)

        mse = mean_squared_error(truths, preds)
        recon_mse = np.mean(recon_errors)

        fold_results.append([corr, mse, recon_mse, epoch_time])

    fold_df = pd.DataFrame(fold_results, columns=['Corr', 'MSE', 'Recon_MSE', 'Time_per_epoch'])
    fold_df.to_csv(os.path.join(variant_dir, f'{trait}_folds.csv'), index=False)

    summary = pd.DataFrame([{
        'trait': trait,
        'corr_mean': fold_df['Corr'].mean(),
        'corr_std': fold_df['Corr'].std(),
        'mse_mean': fold_df['MSE'].mean(),
        'recon_mse_mean': fold_df['Recon_MSE'].mean(),
        'time_per_epoch': fold_df['Time_per_epoch'].mean()
    }])
    summary.to_csv(os.path.join(variant_dir, f'{trait}_summary.csv'), index=False)

    print(f"    {trait}: Corr = {summary['corr_mean'].values[0]:.4f} ± {summary['corr_std'].values[0]:.4f}")

print(f"\n✅ All traits done for variant '{variant_name}'. Results saved in {variant_dir}")