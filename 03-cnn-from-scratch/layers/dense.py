import numpy as np

class Dense:
    """Fully connected layer with He initialization"""
    def __init__(self, input_size, output_size, use_bias=True, rng=None):
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        self.rng = rng if rng is not None else np.random.default_rng(42)
        
        # He initialization for ReLU
        self.W = self.rng.normal(0, np.sqrt(2.0 / input_size), size=(input_size, output_size))
        self.b = np.zeros((1, output_size)) if use_bias else None
        
        self.cache = {}
        
        # Optimizer states
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b) if use_bias else None
        self.v_b = np.zeros_like(self.b) if use_bias else None
    
    def forward(self, X, training=True):
        """Forward pass: X · W + b"""
        self.cache['X'] = X
        Z = np.dot(X, self.W)
        if self.use_bias:
            Z += self.b
        return Z
    
    def backward(self, grad):
        """Backward pass with proper gradient scaling"""
        X = self.cache['X']
        n_samples = X.shape[0]
        
        # Gradients
        dW = np.dot(X.T, grad) / n_samples
        dX = np.dot(grad, self.W.T)  # NO division! (passes total error)
        
        if self.use_bias:
            db = np.sum(grad, axis=0, keepdims=True) / n_samples
        else:
            db = None
        
        return dX, dW, db