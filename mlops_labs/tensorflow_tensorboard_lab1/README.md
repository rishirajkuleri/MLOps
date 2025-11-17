# Neural Network Function Approximation with TensorBoard

A neural network lab focused on using **TensorBoard** for monitoring and visualizing the training process of function approximation models.

## Overview

This lab demonstrates how to leverage TensorBoard to track, visualize, and debug neural network training in real-time. Built a model to approximate mathematical functions while monitoring training metrics, visualizing loss curves, and analyzing model performance through TensorBoard's dashboard.

## TensorBoard Integration

### Logging Setup
```python
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
```

Each training run creates a timestamped directory for organized experiment tracking and comparison.

### Launching TensorBoard
```bash
tensorboard --logdir=logs/scalars
```
Then navigate to `http://localhost:6006` in the browser.

### What TensorBoard Shows You

1. **Scalars Tab**
   - Training loss (MSE) per epoch
   - Validation loss (MSE) per epoch
   - Training MAE per epoch
   - Validation MAE per epoch
   - Learning rate changes over time

2. **Graphs Tab**
   - Complete model architecture visualization
   - Layer connections and tensor shapes
   - Operation-level computation graph

3. **Distributions/Histograms Tab**
   - Weight distributions across layers
   - Bias distributions
   - Activation distributions

### Key Metrics to Monitor

- **Loss curves**
- **MAE vs MSE**
- **Learning rate schedule**
- **Early stopping point**

### Comparing Experiments

TensorBoard allows to:
- Load multiple log directories simultaneously
- Compare different hyperparameter configurations
- Overlay training curves from different runs
- Identify which changes improved performance

## Model Architecture

```
Input (1D) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Output (1D)
```

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: L2 regularization and Dropout layers to prevent overfitting

## Key Components

### Training Callbacks
- **TensorBoard**
- **EarlyStopping**
- **ReduceLROnPlateau**

### Hyperparameters
- Batch size: 32
- Initial learning rate: 0.001
- Max epochs: 50 (with early stopping)
- L2 regularization: 0.001

## Usage

### Quick Start
1. Run all notebook cells to generate data and train the model
2. While training, launch TensorBoard:
   ```bash
   tensorboard --logdir=logs/scalars
   ```
3. Open `http://localhost:6006` to watch training in real-time
4. Explore different tabs to visualize metrics, model graph, and weight distributions

### Experiment Workflow
1. **Baseline run**: Train with default parameters and note the results in TensorBoard
2. **Modify hyperparameters**: Change learning rate, batch size, or architecture
3. **Compare in TensorBoard**: Load both log directories to compare side-by-side
4. **Iterate**: Use insights from TensorBoard to guide further improvements


## Results

The model reports:
- Training and validation MSE
- Training and validation MAE
- Best validation loss achieved
- Total epochs trained before early stopping

## Requirements

- TensorFlow/Keras (includes TensorBoard)
- NumPy
- Matplotlib (for visualization)

### Installing TensorBoard
TensorBoard comes bundled with TensorFlow. If you need to install separately:
```bash
pip install tensorboard
```
