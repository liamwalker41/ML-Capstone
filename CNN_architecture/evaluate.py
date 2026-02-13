import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt  # ✅ for visualization
from config import Config, MODEL_NAME
from load_data import load_data
from train import CNNModel, train_model


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
    Function for evaluating a CNN model.
    """
    print(f"\n--- Evaluating CNN for {dataset_name} ---")
    config = Config()
    dataset_config = getattr(config, dataset_name)

    features, labels = load_data(dataset_name)
    if features is None or labels is None:
        print(f"Could not load data for {dataset_name}. Exiting evaluation.")
        return

    features = features.float()
    labels = labels.long()
    batch_size = config.batch_size

    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"DataLoader created for evaluation with batch_size={batch_size}")

    model.eval()
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
    print("Starting CNN evaluation example...")

    # Train and get history
    cnn_model, training_history = train_model(MODEL_NAME, return_history=True)  # ✅ modify train_model to return history
    if cnn_model:
        eval_accuracy = evaluate_model(MODEL_NAME, cnn_model)
        plot_training_results(training_history, eval_accuracy, MODEL_NAME)

    print("CNN evaluation examples finished.")
