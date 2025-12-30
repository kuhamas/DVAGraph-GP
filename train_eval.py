# train_eval.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from DVAGraphGP import DVAGraphGP
from data_preprocessing import load_data, convert_to_tensor

# Training configuration
batch_size = 32
learning_rate = 1e-4
epochs = 100
weight_decay = 1e-5
latent_dim = 64

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
genotype_file = 'your-genotype-data-path'
phenotype_file = 'your-phenotype-data-path'

X, y = load_data(genotype_file, phenotype_file)
X_tensor, y_tensor = convert_to_tensor(X, y)
print(f"Data shape: X={X_tensor.shape}, y={y_tensor.shape}")

# Initialize the model
model = DVAGraphGP(input_dim=X_tensor.shape[1], latent_dim=latent_dim).to(device)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# K-fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store results
all_predictions = []
all_actuals = []
results = []

# Track total training and evaluation time
total_training_time = 0
total_prediction_time = 0
start_total_time = time.time()

# Training and evaluation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"\nFold {fold + 1} Starting...")

    # Prepare training and validation sets
    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    start_train_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(batch_X)

            # Calculate loss
            loss = loss_fn(y_pred.squeeze(), batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    fold_train_time = time.time() - start_train_time
    total_training_time += fold_train_time
    print(f"Fold {fold + 1} Training Time: {fold_train_time:.2f} seconds")

    # Evaluation loop
    model.eval()
    start_pred_time = time.time()
    y_preds, y_trues = [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            y_out = model(batch_X)
            y_preds.append(y_out.cpu().numpy())
            y_trues.append(batch_y.cpu().numpy())

    fold_pred_time = time.time() - start_pred_time
    total_prediction_time += fold_pred_time
    print(f"Fold {fold + 1} Prediction Time: {fold_pred_time:.2f} seconds")

    # Collect predictions and actual values
    fold_preds = np.concatenate(y_preds)
    fold_actuals = np.concatenate(y_trues)

    # Store predictions for final evaluation
    all_predictions.extend(fold_preds.flatten().tolist())
    all_actuals.extend(fold_actuals.flatten().tolist())

    # Calculate metrics for this fold
    mse = mean_squared_error(fold_actuals, fold_preds)
    mae = mean_absolute_error(fold_actuals, fold_preds)
    r2 = r2_score(fold_actuals, fold_preds)
    corr, _ = pearsonr(fold_actuals.flatten(), fold_preds.flatten())
    results.append([mse, mae, r2, corr])

    print(f"Fold {fold + 1} Results:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Correlation: {corr:.4f}")

# Total training and evaluation time
end_total_time = time.time()
total_execution_time = end_total_time - start_total_time

# Average results
avg_results = np.mean(results, axis=0)

# Report final results
print("\nFinal Evaluation Across All Folds:")
print(f"Average MSE: {avg_results[0]:.4f}")
print(f"Average MAE: {avg_results[1]:.4f}")
print(f"Average R² Score: {avg_results[2]:.4f}")
print(f"Average Correlation: {avg_results[3]:.4f}")

print("\nTime Statistics:")
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Total Prediction Time: {total_prediction_time:.2f} seconds")
print(f"Total Execution Time: {total_execution_time:.2f} seconds")

# Save results to CSV
import pandas as pd
results_df = pd.DataFrame(results, columns=['MSE', 'MAE', 'R2', 'Correlation'])
results_df['Fold'] = range(1, len(results) + 1)
results_df.to_csv('training_evaluation_results.csv', index=False)
print("\nSaved detailed results to 'training_evaluation_results.csv'")
