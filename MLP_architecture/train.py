import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import Config, MODEL_NAME
from load_data import load_data

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers, dropout_rate=0.0):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(dataset_name, return_history=False):
    """
    Function for training an MLP model.
    
    Args:
        dataset_name: Name of the dataset to train on
        return_history: If True, returns (model, history_dict), otherwise just returns model
    """
    print(f"\n--- Training MLP for {dataset_name} ---")
    config = Config()
    dataset_config = getattr(config, dataset_name)

    features, labels = load_data(dataset_name)
    if features is None or labels is None:
        print(f"Could not load data for {dataset_name}. Exiting training.")
        return None if not return_history else (None, None)

    # Ensure features are float and labels are long
    features = features.float()
    labels = labels.long()

    # Dynamically determine input_dim from the loaded features
    actual_input_dim = features.shape[1]

    num_classes = dataset_config['num_classes']
    hidden_layers = dataset_config['hidden_layers']
    dropout_rate = config.dropout_rate
    batch_size = config.batch_size

    model = MLPModel(actual_input_dim, num_classes, hidden_layers, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Model initialized with input_dim={actual_input_dim}, num_classes={num_classes}, hidden_layers={hidden_layers}, dropout_rate={dropout_rate}")
    print(f"Optimizer: {optimizer.__class__.__name__}, Learning Rate: {config.learning_rate}")
    print(f"Loss Function: {criterion.__class__.__name__}")

    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with batch_size={batch_size}")

    # Initialize training history
    training_history = {'loss': []}

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_features, batch_labels) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        training_history['loss'].append(avg_epoch_loss)
        
        print(f'Epoch [{epoch+1}/{config.epochs}], Loss: {avg_epoch_loss:.4f}')

    print(f"Training for {dataset_name} complete.")
    
    if return_history:
        return model, training_history
    else:
        return model

if __name__ == '__main__':
    print("Starting MLP training example...")
    mlp_model = train_model(MODEL_NAME)
    print("MLP training examples finished.")
