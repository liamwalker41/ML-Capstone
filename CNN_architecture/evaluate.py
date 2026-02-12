import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader # Import TensorDataset and DataLoader
from config import Config, MODEL_NAME
from load_data import load_data
from train import CNNModel, train_model # Import CNNModel and train_model from train.py

def evaluate_model(dataset_name, model):
    """
    Function for evaluating a CNN model.
    """
    print(f"\n--- Evaluating CNN for {dataset_name} ---")
    config = Config()
    dataset_config = getattr(config, dataset_name)

    features, labels = load_data(dataset_name)
    if features is None or labels is None:
        print(f"Could not load data for {dataset_name}. Exiting evaluation.")
        return

    # Ensure features are float and labels are long
    features = features.float()
    labels = labels.long()
    batch_size = config.batch_size # Get batch size from config

    # Create DataLoader for evaluation
    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # No shuffle for evaluation
    print(f"DataLoader created for evaluation with batch_size={batch_size}")

    # Evaluation logic
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Evaluation for {dataset_name} complete. Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    # Example Usage:
    print("Starting CNN evaluation example...")

    # Train and evaluate UCI Adult
    cnn_model = train_model(MODEL_NAME)
    if cnn_model: # Only evaluate if training was successful
        evaluate_model(MODEL_NAME, cnn_model)

    print("CNN evaluation examples finished.")