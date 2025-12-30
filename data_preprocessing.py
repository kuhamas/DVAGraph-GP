# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_data(genotype_file, phenotype_file):
    """
    Load genotype and phenotype data from CSV files.

    Args:
    - genotype_file: path to the genotype data CSV file
    - phenotype_file: path to the phenotype data CSV file

    Returns:
    - X: Preprocessed genotype data (features)
    - y: Target phenotype values
    """
    # Step 1: Load Genotype Data
    genotype = pd.read_csv(genotype_file)

    # Step 2: Drop the first column (assuming it's an ID or non-numeric column)
    X_raw = genotype.drop(genotype.columns[0], axis=1).values

    # Step 3: Normalize the genotype data using MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    # Step 4: Load Target Data (Phenotype Data)
    targets = pd.read_csv(phenotype_file)

    # Step 5: Extract the 'SYPP' column for the target values (adjust to your specific column)
    y = targets['SYPP'].values  # Replace 'SYPP' with the correct column name

    return X, y


def convert_to_tensor(X, y):
    """
    Convert NumPy arrays to PyTorch tensors and move to device.

    Args:
    - X: Feature data (genotype)
    - y: Target data (phenotype)

    Returns:
    - X_tensor: PyTorch tensor for features
    - y_tensor: PyTorch tensor for targets
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)  # Ensure it's a column vector

    return X_tensor, y_tensor


def save_processed_data(X, y, X_file, y_file):
    """
    Save the preprocessed data to CSV files.

    Args:
    - X: Feature data
    - y: Target data
    - X_file: Path to save the preprocessed features
    - y_file: Path to save the preprocessed targets
    """
    pd.DataFrame(X).to_csv(X_file, index=False)
    pd.DataFrame(y).to_csv(y_file, index=False)
    print(f"Data saved: {X_file}, {y_file}")


if __name__ == "__main__":
    # Define file paths (Update with your actual file paths)
    genotype_file = 'genotype-data-path'
    phenotype_file = 'phenotype-data-path'

    # Load data
    X, y = load_data(genotype_file, phenotype_file)

    # Convert data to tensors
    X_tensor, y_tensor = convert_to_tensor(X, y)

    # Save processed data (optional)
    save_processed_data(X, y, 'processed_genotype.csv', 'processed_phenotype.csv')
