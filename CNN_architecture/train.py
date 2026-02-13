import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # Import TensorDataset and DataLoader
from config import Config, MODEL_NAME
from load_data import load_data

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes, cnn_layers=None, dropout_rate=0.0):
        super(CNNModel, self).__init__()

        if len(input_shape) == 2: # UCI Adult: (channels, length) for 1D CNN
            in_channels = input_shape[0]

            feature_extractor_layers = []
            feature_extractor_layers.append(nn.Conv1d(in_channels, 32, kernel_size=3, padding=1))
            feature_extractor_layers.append(nn.ReLU())
            if dropout_rate > 0: feature_extractor_layers.append(nn.Dropout(p=dropout_rate))
            feature_extractor_layers.append(nn.MaxPool1d(kernel_size=2))
            feature_extractor_layers.append(nn.Conv1d(32, 64, kernel_size=3, padding=1))
            feature_extractor_layers.append(nn.ReLU())
            if dropout_rate > 0: feature_extractor_layers.append(nn.Dropout(p=dropout_rate))
            feature_extractor_layers.append(nn.MaxPool1d(kernel_size=2))

            self.feature_extractor = nn.Sequential(*feature_extractor_layers)

            dummy_input = torch.zeros(1, *input_shape)
            with torch.no_grad():
                dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

            classifier_layers = [
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
            ]
            if dropout_rate > 0:
                classifier_layers.append(nn.Dropout(p=dropout_rate))
            classifier_layers.append(nn.Linear(128, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)

        elif len(input_shape) == 3: # CIFAR-100, PCam: (channels, height, width) for 2D CNN
            in_channels = input_shape[0]
            feature_extractor_layers = []
            current_channels = in_channels

            if cnn_layers:
                for i, layer_params in enumerate(cnn_layers):
                    feature_extractor_layers.append(nn.Conv2d(current_channels, layer_params['filters'], kernel_size=layer_params['kernel_size'], padding=(layer_params['kernel_size'] // 2)))
                    feature_extractor_layers.append(nn.ReLU())
                    if dropout_rate > 0: feature_extractor_layers.append(nn.Dropout(p=dropout_rate))
                    if 'pool_size' in layer_params:
                        feature_extractor_layers.append(nn.MaxPool2d(kernel_size=layer_params['pool_size']))
                    current_channels = layer_params['filters']
            else:
                # Default 2D CNN if cnn_layers is not provided (e.g., fallback)
                feature_extractor_layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
                feature_extractor_layers.append(nn.ReLU())
                if dropout_rate > 0: feature_extractor_layers.append(nn.Dropout(p=dropout_rate))
                feature_extractor_layers.append(nn.MaxPool2d(kernel_size=2))
                feature_extractor_layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
                feature_extractor_layers.append(nn.ReLU())
                if dropout_rate > 0: feature_extractor_layers.append(nn.Dropout(p=dropout_rate))
                feature_extractor_layers.append(nn.MaxPool2d(kernel_size=2))
                current_channels = 64

            self.feature_extractor = nn.Sequential(*feature_extractor_layers)

            dummy_input = torch.zeros(1, *input_shape)
            with torch.no_grad():
                dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

            classifier_layers = [
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
            ]
            if dropout_rate > 0:
                classifier_layers.append(nn.Dropout(p=dropout_rate))
            classifier_layers.append(nn.Linear(128, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)

        else:
            raise ValueError("Unsupported input_shape for CNNModel")

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) # Flatten for the classifier
        x = self.classifier(x)
        return x

def train_model(dataset_name, return_history=False):
    """
    Function for training a CNN model.
    
    Args:
        dataset_name: Name of the dataset to train on
        return_history: If True, returns (model, history_dict), otherwise just returns model
    """
    print(f"\n--- Training CNN for {dataset_name} ---")
    config = Config()
    dataset_config = getattr(config, dataset_name)

    features, labels = load_data(dataset_name)
    if features is None or labels is None:
        print(f"Could not load data for {dataset_name}. Exiting training.")
        return None if not return_history else (None, None)

    # Ensure features are float and labels are long
    features = features.float()
    labels = labels.long()

    # Dynamically determine input_shape from the loaded features
    if dataset_name == 'UCI_Adult':
        # For UCI Adult, features shape is (num_samples, channels, length)
        # So input_shape for CNNModel should be (channels, length)
        actual_input_shape = (features.shape[1], features.shape[2])
    else:
        # For image datasets (CIFAR-100, PCam), features shape is (num_samples, channels, height, width)
        # So input_shape for CNNModel should be (channels, height, width)
        actual_input_shape = (features.shape[1], features.shape[2], features.shape[3])

    num_classes = dataset_config['num_classes']
    cnn_layers = dataset_config.get('cnn_layers') # Will be None for UCI Adult
    dropout_rate = config.dropout_rate # Get dropout rate from config
    batch_size = config.batch_size # Get batch size from config

    model = CNNModel(actual_input_shape, num_classes, cnn_layers, dropout_rate) # Pass dropout_rate to model
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Model initialized with input_shape={actual_input_shape}, num_classes={num_classes}, dropout_rate={dropout_rate}")
    print(f"Optimizer: {optimizer.__class__.__name__}, Learning Rate: {config.learning_rate}")
    print(f"Loss Function: {criterion.__class__.__name__}")

    # Create DataLoader
    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with batch_size={batch_size}")

    # Initialize training history
    training_history = {'loss': []}

    # Realistic training loop
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
    print("Starting CNN training example...")
    cnn_model = train_model(MODEL_NAME)
    print("CNN training example finished.")
