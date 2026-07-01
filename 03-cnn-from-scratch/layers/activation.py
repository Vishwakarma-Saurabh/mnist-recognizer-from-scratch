import numpy as np

class Activation:
    """Activation layer with multiple activation functions"""
    def __init__(self, activation_type='relu'):
        self.activation_type = activation_type
        self.cache = {}
    
    def forward(self, X, training=True):
        """Forward pass: Apply activation function"""
        self.cache['X'] = X
        
        if self.activation_type == 'relu':
            return np.maximum(0, X)
        elif self.activation_type == 'leaky_relu':
            return np.where(X > 0, X, 0.01 * X)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        elif self.activation_type == 'softmax':
            X_stable = X - np.max(X, axis=1, keepdims=True)
            exp_X = np.exp(X_stable)
            return exp_X / np.sum(exp_X, axis=1, keepdims=True)
        elif self.activation_type == 'tanh':
            return np.tanh(X)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
    
    def backward(self, grad):
        """Backward pass: Apply derivative of activation"""
        X = self.cache['X']
        
        if self.activation_type == 'relu':
            return grad * (X > 0).astype(float)
        elif self.activation_type == 'leaky_relu':
            return grad * np.where(X > 0, 1.0, 0.01)
        elif self.activation_type == 'sigmoid':
            sig = 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
            return grad * sig * (1 - sig)
        elif self.activation_type == 'softmax':
            return grad  # Combined with cross-entropy loss
        elif self.activation_type == 'tanh':
            return grad * (1 - np.tanh(X) ** 2)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")

class ReLU(Activation):
    """ReLU activation layer"""
    def __init__(self):
        super().__init__('relu')

class Softmax(Activation):
    """Softmax activation layer"""
    def __init__(self):
        super().__init__('softmax')