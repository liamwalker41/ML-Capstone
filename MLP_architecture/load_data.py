import os
import pandas as pd
import numpy as np
import torch
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# For CIFAR-100
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config

# --- UCI Adult Data Loading and Preprocessing ---
UCI_ADULT_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

def download_and_load_uci_adult(url, column_names):
    response = requests.get(url)
    response.raise_for_status # Raise an exception for HTTP errors
    data = io.StringIO(response.text)
    df = pd.read_csv(data, header=None, names=column_names, na_values=[' ?']) # Handle ' ?' as NaN
    return df

def load_uci_adult_data(config):
    print("Downloading and loading UCI Adult data...")
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    df_train = download_and_load_uci_adult(train_url, UCI_ADULT_COLUMNS)
    # The test file has an extra dot at the beginning of the lines
    df_test = download_and_load_uci_adult(test_url, UCI_ADULT_COLUMNS).iloc[1:] # Drop the first row which is often metadata/empty

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop rows with NaN values (from ' ?' in original data)
    df.dropna(inplace=True)

    # Separate features (X) and target (y)
    X = df.drop('income', axis=1)
    y = df['income']

    # Preprocessing
    # Target variable encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'}))

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include='object').columns

    # Explicitly convert categorical columns to string type to avoid mixed types
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    # Scale numerical features
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X[numerical_cols])

    # One-hot encode categorical features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_categorical_encoded = ohe.fit_transform(X[categorical_cols])

    # Combine preprocessed features
    X_processed = np.hstack((X_numerical_scaled, X_categorical_encoded))

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(X_processed, dtype=torch.float32)
    labels_tensor = torch.tensor(y_encoded, dtype=torch.long)

    print(f"UCI Adult data loaded and preprocessed. Features shape: {features_tensor.shape}, Labels shape: {labels_tensor.shape}")
    return features_tensor, labels_tensor

# --- CIFAR-100 Data Loading and Preprocessing ---
def load_cifar_100_data(config):
    print("Loading and preprocessing CIFAR-100 data...")
    # Define transformations: resize, convert to tensor, normalize, and flatten
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize images
        transforms.Lambda(lambda x: x.view(-1)) # Flatten for MLP
    ])

    # Load CIFAR-100 training dataset
    cifar100_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # Extract features and labels
    features_list = []
    labels_list = []

    # Using a DataLoader to iterate over the dataset
    # batch_size set to a high number to get all data at once for this example
    data_loader = DataLoader(cifar100_dataset, batch_size=len(cifar100_dataset), shuffle=False)

    for batch_features, batch_labels in data_loader:
        features_list.append(batch_features)
        labels_list.append(batch_labels)

    features_tensor = torch.cat(features_list, dim=0) if features_list else torch.empty(0)
    labels_tensor = torch.cat(labels_list, dim=0) if labels_list else torch.empty(0)

    print(f"CIFAR-100 data loaded and preprocessed. Features shape: {features_tensor.shape}, Labels shape: {labels_tensor.shape}")
    return features_tensor, labels_tensor

# --- Main load_data function ---
def load_data(dataset_name):
    """
    Loads and preprocesses data for the specified dataset, preparing it as flattened tensors
    suitable for an MLP.

    Args:
        dataset_name (str): The name of the dataset to load ('UCI_Adult', 'CIFAR_100', 'PCam').

    Returns:
        tuple: A tuple containing (features_tensor, labels_tensor).
               Returns (None, None) if the dataset_name is not recognized.
    """
    config = Config()

    if dataset_name == 'UCI_Adult':
        return load_uci_adult_data(config)

    elif dataset_name == 'CIFAR_100':
        return load_cifar_100_data(config)

    elif dataset_name == 'PCam':
        print(f"Loading and preprocessing data for {dataset_name}...")
        dataset_config = config.PCam

        # Due to the large size of the PCam dataset, retaining placeholder data generation.
        input_dim = dataset_config['input_dim']
        num_classes = dataset_config['num_classes']
        # Example: 1000 samples for features and labels
        features = torch.rand(1000, input_dim) # Placeholder: Flattened image tensor
        labels = torch.randint(0, num_classes, (1000,)) # Placeholder: Label tensor

        print(f"Loaded {dataset_name} data with features shape: {features.shape}, Labels shape: {labels.shape} (Placeholder)")
        return features, labels

    else:
        print(f"Error: Unknown dataset_name '{dataset_name}'")
        return None, None

# Example Usage:
if __name__ == '__main__':
    print("\n--- Example Usage ---")

    # Load UCI Adult data
    uci_features, uci_labels = load_data('UCI_Adult')
    if uci_features is not None:
        print(f"UCI Adult Features type: {uci_features.dtype}, Labels type: {uci_labels.dtype}")

    print("\n")

    # Load CIFAR-100 data
    cifar_features, cifar_labels = load_data('CIFAR_100')
    if cifar_features is not None:
        print(f"CIFAR-100 Features type: {cifar_features.dtype}, Labels type: {cifar_labels.dtype}")

    print("\n")

    # Load PCam data (placeholder)
    pcam_features, pcam_labels = load_data('PCam')
    if pcam_features is not None:
        print(f"PCam Features type: {pcam_features.dtype}, Labels type: {pcam_labels.dtype}")

    print("\n--- End Example Usage ---")