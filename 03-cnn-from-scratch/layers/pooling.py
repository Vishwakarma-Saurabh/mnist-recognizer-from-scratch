import numpy as np

class MaxPool2D:
    """Max Pooling layer from scratch"""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.max_positions = {}
        self.cache = {}
    
    def forward(self, X, training=True):
        """Forward pass: Take maximum in each window"""
        batch_size, channels, in_h, in_w = X.shape
        
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_positions = {}
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        window = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        output[b, c, i, j] = max_val
                        
                        # Store position of max for backprop
                        max_idx = np.argmax(window)
                        max_pos = np.unravel_index(max_idx, window.shape)
                        self.max_positions[(b, c, i, j)] = (
                            h_start + max_pos[0],
                            w_start + max_pos[1]
                        )
        
        self.cache['X'] = X
        self.cache['output'] = output
        return output
    
    def backward(self, grad_output):
        """Backward pass: Route gradients to max positions"""
        X = self.cache['X']
        batch_size, channels, in_h, in_w = X.shape
        
        dX = np.zeros_like(X)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        # Get position of max
                        h_pos, w_pos = self.max_positions[(b, c, i, j)]
                        # Route gradient to max position
                        dX[b, c, h_pos, w_pos] += grad_output[b, c, i, j]
        
        return dX

class AveragePool2D:
    """Average Pooling layer from scratch"""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, X, training=True):
        """Forward pass: Take average in each window"""
        batch_size, channels, in_h, in_w = X.shape
        
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        window = X[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.mean(window)
        
        self.cache['X'] = X
        self.cache['output'] = output
        return output
    
    def backward(self, grad_output):
        """Backward pass: Distribute gradients evenly"""
        X = self.cache['X']
        batch_size, channels, in_h, in_w = X.shape
        
        dX = np.zeros_like(X)
        pool_size_float = float(self.pool_size ** 2)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Distribute gradient evenly across the window
                        dX[b, c, h_start:h_end, w_start:w_end] += \
                            grad_output[b, c, i, j] / pool_size_float
        
        return dX