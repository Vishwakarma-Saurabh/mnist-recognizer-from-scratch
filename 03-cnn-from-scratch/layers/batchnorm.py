import numpy as np

class BatchNorm2D:
    """Batch Normalization for 2D inputs (convolutional layers)"""
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters (shape: 1, features, 1, 1 for broadcasting)
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        self.cache = {}
        
        # Optimizer states
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)
        self.v_beta = np.zeros_like(self.beta)
    
    def forward(self, X, training=True):
        """Forward pass: Normalize and scale/shift"""
        if training:
            # Calculate batch statistics
            mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            var = np.var(X, axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalize
            X_norm = (X - mean) / np.sqrt(var + self.eps)
            self.cache = {'mean': mean, 'var': var, 'X_norm': X_norm, 'training': True}
        else:
            # Use running statistics
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.cache = {'training': False}
        
        # Scale and shift
        return self.gamma * X_norm + self.beta
    
    def backward(self, grad):
        """Backward pass with correct BatchNorm gradient"""
        if not self.cache.get('training', False):
            # No backward during inference
            return grad
        
        X = self.cache['X_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        n = grad.shape[0] * grad.shape[2] * grad.shape[3]  # Total spatial positions
        
        # Gradient w.r.t gamma and beta
        dgamma = np.sum(grad * X, axis=(0, 2, 3), keepdims=True) / n
        dbeta = np.sum(grad, axis=(0, 2, 3), keepdims=True) / n
        
        # Gradient w.r.t input
        dX_norm = grad * self.gamma
        
        # Formula: dX = (1/N) * (dX_norm - mean(dX_norm) - X_norm * mean(dX_norm * X_norm))
        dX = (1 / n) * (
            dX_norm - 
            np.mean(dX_norm, axis=(0, 2, 3), keepdims=True) - 
            X * np.mean(dX_norm * X, axis=(0, 2, 3), keepdims=True)
        )
        
        dX = dX / np.sqrt(var + self.eps)
        
        return dX, dgamma, dbeta