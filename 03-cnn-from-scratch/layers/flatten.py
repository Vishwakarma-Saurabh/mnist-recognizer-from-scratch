import numpy as np

class Flatten:
    """Flatten layer: Convert multi-dimensional to 1D"""
    def __init__(self):
        self.cache = {}
    
    def forward(self, X, training=True):
        """Forward pass: Flatten all dimensions except batch"""
        self.cache['shape'] = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, grad_output):
        """Backward pass: Reshape back to original shape"""
        return grad_output.reshape(self.cache['shape'])