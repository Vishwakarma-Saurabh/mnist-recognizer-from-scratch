import numpy as np

class Dropout:
    """Dropout regularization layer"""
    def __init__(self, rate=0.5, rng=None):
        self.rate = rate
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.mask = None
        self.cache = {}
    
    def forward(self, X, training=True):
        """Forward pass with dropout during training only"""
        if training and self.rate > 0:
            # Create mask and scale
            self.mask = (self.rng.random(X.shape) > self.rate) / (1 - self.rate)
            return X * self.mask
        else:
            self.mask = None
            return X
    
    def backward(self, grad):
        """Backward pass: Apply same mask"""
        if self.mask is not None:
            return grad * self.mask
        return grad