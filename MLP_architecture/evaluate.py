import torch
import torch.nn as nn # Not strictly needed for evaluation but good practice
from torch.utils.data import TensorDataset, DataLoader # Import TensorDataset and DataLoader
import matplotlib.pyplot as plt
from config import Config, MODEL_NAME
from load_data import load_data
from train import MLPModel, train_model # Import MLPModel and train_model from train.py


def plot_training_results(training_history, eval_accuracy, dataset_name):
    """
    Plot training loss and evaluation accuracy.
    """
    epochs = range(1, len(training_history['loss']) + 1)

    plt.figure(figsize=(10, 5))

    # Plot 1: Training Loss per Epoch
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_history['loss'], 'b-o', label='Training Loss')
    plt.title(f'{dataset_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Evaluation Accuracy
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy'], [eval_accuracy], color='green')
    plt.title(f'{dataset_name} - Evaluation Accuracy')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()


def evaluate_model(dataset_name, model):
    """
    Function for evaluating an MLP model.
    """
    print(f"\n--- Evaluating MLP for {dataset_name} ---")
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
    return accuracy


if __name__ == '__main__':
    # Example Usage:
    print("Starting MLP evaluation example...")

    # Train and evaluate with visualization
    print(f"\n--- Training {MODEL_NAME} model for evaluation ---")
    mlp_model, training_history = train_model(MODEL_NAME, return_history=True)
    if mlp_model: # Only evaluate if training was successful
        eval_accuracy = evaluate_model(MODEL_NAME, mlp_model)
        plot_training_results(training_history, eval_accuracy, MODEL_NAME)

    print("MLP evaluation examples finished.")
