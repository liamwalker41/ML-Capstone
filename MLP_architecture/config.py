MODEL_NAME = 'PCam'  # Change this to 'UCI_Adult', 'CIFAR_100' or 'PCam' to train on different datasets
class Config:
    def __init__(self):
        # General training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.dropout_rate = 0.5 # Added dropout rate for regularization in MLP

        # Dataset-specific configurations
        self.UCI_Adult = {
            'data_path': 'data/uci_adult/',  # Path
            'num_features': 108,               # Example based on one-hot encoding
            'num_classes': 2,                  # Binary classification
            'hidden_layers': [256, 128, 64]    # Example MLP architecture
        }

        self.CIFAR_100 = {
            'data_path': 'data/cifar_100/',    # Path
            'image_size': 32,                  # 32x32 images
            'image_channels': 3,               # RGB channels
            'input_dim': 32 * 32 * 3,          # Flattened input for MLP
            'num_classes': 100,                # 100 classes
            'hidden_layers': [1024, 512, 256]  # Example MLP architecture
        }

        self.PCam = {
            'data_path': 'data/pcam/',         # Path
            'image_size': 96,                  # 96x96 images
            'image_channels': 3,               # RGB channels
            'input_dim': 96 * 96 * 3,          # Flattened input for MLP
            'num_classes': 2,                  # Binary classification
            'hidden_layers': [2048, 1024, 512] # Example MLP architecture
        }

# Example usage:
# config = Config()
# print(config.UCI_Adult['hidden_layers'])
# print(config.CIFAR_100['input_dim'])
