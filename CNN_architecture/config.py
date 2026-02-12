MODEL_NAME ='CIFAR_100' # Change this to 'UCI_Adult', 'CIFAR_100', or 'PCam' to select dataset and model configuration
class Config:
    def __init__(self):
        # General training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.dropout_rate = 0.5 # Added dropout rate for CNN

        # Dataset-specific configurations
        self.UCI_Adult = {
            'data_path': 'data/uci_adult/',  # Path
            'num_features': 108,               # Example based on one-hot encoding
            'num_classes': 2,                  # Binary classification
            'input_shape': (1, 108)            # For 1D CNN: (channels, length)
        }

        self.CIFAR_100 = {
            'data_path': 'data/cifar_100/',    # Path
            'image_size': 32,                  # 32x32 images
            'image_channels': 3,               # RGB channels
            'num_classes': 100,                # 100 classes
            'input_shape': (3, 32, 32),        # For 2D CNN: (channels, height, width)
            'cnn_layers': [
                {'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
                {'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2}
            ]
        }

        self.PCam = {
            'data_path': 'data/pcam/',         # Path
            'image_size': 96,                  # 96x96 images
            'image_channels': 3,               # RGB channels
            'num_classes': 2,                  # Binary classification
            'input_shape': (3, 96, 96),        # For 2D CNN: (channels, height, width)
            'cnn_layers': [
                {'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
                {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
                {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2}
            ]
        }

# Example usage:
# config = Config()
# print(config.CIFAR_100['input_shape'])
# print(config.PCam['cnn_layers'][0]['filters'])
# print(config.dropout_rate)