# PyTorch CNN & MLP Training and Evaluation

A PyTorch-based project for training and evaluating Convolutional Neural Networks (CNNs) and Multi-layer Perceptrons (MLPs) on multiple datasets including UCI Adult, CIFAR-100, and PCam.

## Features

- **Flexible CNN Architecture**: Supports both 1D CNNs (for tabular data) and 2D CNNs (for image data)
- **Flexible MLP Architecture**: Supports preparing data as flattened tensors suitable for an MLP.
- **Multiple Dataset Support**: Compatible with UCI Adult, CIFAR-100, and PCam datasets
- **Training Visualization**: Plots training loss curves and evaluation accuracy
- **Configurable Hyperparameters**: Easy-to-adjust learning rate, batch size, dropout rate, and epochs
- **Model Evaluation**: Comprehensive evaluation metrics with accuracy reporting

## Installation

### Prerequisites

Python 3.7 or higher is required.

### Install Dependencies

Install PyTorch and related packages:

```bash
pip install torch torchvision
```

Install matplotlib for visualization:

```bash
pip install matplotlib
```

Install pandas:
```bash
pip install pandas
```

Install requests:
```bash
pip install requests
```

Install sklearn:
```bash
pip install scikit-learn
```

Or install all dependencies at once:

```bash
pip install torch torchvision matplotlib pandas requests scikit-learn
```

## Project Structure

```
.
├── CNN_architecture
    ├── train.py           # Training script with CNN model definition
    ├── evaluate.py        # Evaluation script with visualization
    ├── load_data.py       # Data loading utilities
    ├── config.py          # Configuration file for hyperparameters
├── MLP_architecture
    ├── train.py           # Training script with MLP model definition
    ├── evaluate.py        # Evaluation script with visualization
    ├── load_data.py       # Data loading utilities
    ├── config.py          # Configuration file for hyperparameters
├── data                   # Path to store data
└── README.md          # This file
```

## Usage

### Training a Model

To train a model on the default dataset, navigate to the directory for your desired architecture (CNN_architecture or MLP_architecture) and run:

```bash
python train.py
```

### Training and Evaluating with Visualization

To train a model, evaluate it, and visualize the results:

```bash
python evaluate.py
```

This will:
1. Train the model on the specified dataset
2. Evaluate the model's accuracy
3. Display plots showing training loss per epoch and final evaluation accuracy

## Configuration

Edit `config.py` to adjust hyperparameters:

- `learning_rate`: Learning rate for the optimizer (default: 0.001)
- `batch_size`: Batch size for training (default: 32)
- `epochs`: Number of training epochs (default: 10)
- `dropout_rate`: Dropout rate for regularization (default: 0.0)

### Dataset Configuration

The `Config` class in `config.py` should define dataset-specific settings:

```python
MODEL_NAME = 'CIFAR_100'  # or 'UCI_Adult', 'PCam'
```

Each dataset configuration includes:
- `num_classes`: Number of output classes
- `cnn_layers`: CNN architecture specification (for 2D CNNs)

## Model Architecture

### CNNModel Class

The `CNNModel` class automatically adapts to the input data:

- **1D CNN** (for UCI Adult dataset): Uses Conv1d layers for sequential/tabular data
- **2D CNN** (for image datasets): Uses Conv2d layers with customizable architecture

Features:
- Configurable dropout for regularization
- Dynamic input shape detection
- Feature extraction followed by classification layers

### MLPModel Class

The `MLPModel` class automatically adapts to the input data:

- **Layers & Dimensions** Dynamically adds appropriate layers and dimensions based on input data

Features:
- Dynamic input shape detection
- Dynamically creates layers

## Training History

The training process tracks:
- Average loss per epoch
- Can be extended to track additional metrics (accuracy, validation loss, etc.)

## Output

After running `evaluate.py`, you'll see:
- Console output with training progress and final accuracy
- Two plots:
  - Training loss over epochs
  - Final evaluation accuracy

## Example Output

```
--- Training CNN for CIFAR_100 ---
Model initialized with input_shape=(3, 32, 32), num_classes=100, dropout_rate=0.0
DataLoader created with batch_size=32
Epoch [1/10], Loss: 4.2345
Epoch [2/10], Loss: 3.8901
...
Training for CIFAR_100 complete.

--- Evaluating CNN for CIFAR_100 ---
Evaluation for CIFAR_100 complete. Accuracy: 85.23%
```

## Extending the Project

### Adding New Datasets

1. Add dataset loading logic to `load_data.py`
2. Add dataset configuration to `config.py`
3. Update `MODEL_NAME` in `config.py`

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
