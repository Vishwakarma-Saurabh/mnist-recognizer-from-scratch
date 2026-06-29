import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class DataLoader:
    """Enhanced DataLoader with proper preprocessing"""
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.input_size = None
        
    def load_mnist(self, test_size=0.2, val_size=0.1):

        print("📂 Loading MNIST dataset...")
        
        # Fetch MNIST data from OpenML
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
        
        # Convert to numpy arrays and normalize pixel values
        X = X.astype(np.float32) / 255.0  # Normalize to [0,1]
        y = y.astype(np.int32)
        
        # Store input size for network configuration
        self.input_size = X.shape[1]
        
        # First split: Training + Validation vs Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: Training vs Validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        # Store data
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Validation set: {X_val.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        return self
    
    def get_batch(self, X, y, batch_size, shuffle=True):
        n_samples = X.shape[0]
        
        if shuffle:
            if hasattr(self, 'rng') and self.rng is not None:
                indices = self.rng.permutation(n_samples)
            else:
                indices = np.random.default_rng(42).permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def one_hot_encode(self, y, num_classes=10):

        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, num_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def get_stats(self):

        return {
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'input_size': self.input_size,
            'num_classes': 10,
            'class_distribution': {
                'train': np.bincount(self.y_train) if self.y_train is not None else None,
                'val': np.bincount(self.y_val) if self.y_val is not None else None,
                'test': np.bincount(self.y_test) if self.y_test is not None else None
            }
        }