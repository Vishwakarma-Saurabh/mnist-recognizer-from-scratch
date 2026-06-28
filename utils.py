import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.input_size = None
        
    def load_mnist(self, test_size=0.2, val_size=0.1):
        print("Loading MNIST dataset...")
        
        # Fetch MNIST data from OpenML
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        # Convert to numpy arrays and normalize pixel values
        X = X.astype(np.float32) / 255.0  # Normalize to [0,1]
        y = y.astype(np.int32)
        
        # Store input size for network configuration
        self.input_size = X.shape[1]
        
        # Split: Training + Validation + Test
        # First split: Test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: Training (70%) and Validation (10%)
        val_ratio = val_size / (1 - test_size)  # Calculate validation ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        # Store data
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return self
    
    def get_batch(self, X, y, batch_size):
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)  # Shuffle
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def one_hot_encode(self, y, num_classes=10):
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, num_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

class Visualizer:
    @staticmethod
    def show_sample_images(X, y, num_samples=10):

        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = X[i].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {y[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_weights(weights, title="Learned Weights"):
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(32, weights.shape[1])):
            # Reshape weights for neuron i to 28x28
            w = weights[:, i].reshape(28, 28)
            axes[i].imshow(w, cmap='RdBu_r')
            axes[i].axis('off')
            axes[i].set_title(f'Neuron {i+1}')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(history):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()