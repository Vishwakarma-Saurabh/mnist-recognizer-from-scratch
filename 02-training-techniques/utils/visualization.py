import numpy as np
import matplotlib.pyplot as plt
from models.layers import Dense

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        """Plot loss and accuracy curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(history['lr'], color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time plot
        axes[1, 1].plot(np.cumsum(history['time']), label='Cumulative Time', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Training Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_lr_schedule(history):
        """Plot learning rate schedule separately"""
        plt.figure(figsize=(10, 4))
        plt.plot(history['lr'], color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule Over Time')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_weights(model, num_neurons=16):
        """Plot learned weights safely"""
        # Get first hidden layer weights
        weights = None
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights = layer.W
                break
        
        if weights is None:
            print("No Dense layer found to visualize.")
            return
            
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.ravel()
        
        for i in range(min(num_neurons, weights.shape[1])):
            w = weights[:, i].reshape(28, 28)
            axes[i].imshow(w, cmap='RdBu_r')
            axes[i].axis('off')
            axes[i].set_title(f'N{i+1}')
        
        plt.suptitle('Hidden Layer Weights', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def show_predictions(model, X, y, num_samples=12):
        """Show sample predictions matching the isolated pipeline seed engine"""
        predictions = model.predict(X)
        
        # FIX: Extract the secure seed generator directly from the model wrapper
        generator = model.rng if hasattr(model, 'rng') and model.rng is not None else np.random.default_rng(42)
        indices = generator.choice(len(X), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            img = X[idx].reshape(28, 28)
            true_label = y[idx]
            pred_label = predictions[idx]
            confidence = np.max(model.forward(X[idx:idx+1], training=False))
            
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}',
                color=color
            )
        
        plt.tight_layout()
        plt.show()