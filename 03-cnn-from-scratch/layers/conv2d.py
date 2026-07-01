import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, 
                 padding=0, stride=1, rng=None):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        
        # Handle kernel_size as tuple
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        # RNG for reproducibility
        self.rng = rng if rng is not None else np.random.default_rng(42)
        
        # He initialization for ReLU
        self.kernels = self.rng.normal(
            0, 
            np.sqrt(2.0 / (in_channels * self.kernel_h * self.kernel_w)),
            size=(out_channels, in_channels, self.kernel_h, self.kernel_w)
        )
        self.bias = np.zeros(out_channels)
        
        # Cache for backward pass
        self.cache = {}
        
        # Optimizer states (for Adam)
        self.m_kernels = np.zeros_like(self.kernels)
        self.v_kernels = np.zeros_like(self.kernels)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
    
    def forward(self, X, training=True):
        batch_size, in_channels, in_h, in_w = X.shape
        
        # Store input for backward
        self.cache['X'] = X
        
        # Apply padding if needed
        if self.padding > 0:
            X_padded = np.pad(
                X, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            X_padded = X
        
        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_w) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(in_channels):
                    # Extract kernel for this output and input channel
                    kernel = self.kernels[oc, ic]
                    
                    # Slide kernel over input
                    for i in range(out_h):
                        for j in range(out_w):
                            # Get current window
                            h_start = i * self.stride
                            h_end = h_start + self.kernel_h
                            w_start = j * self.stride
                            w_end = w_start + self.kernel_w
                            
                            window = X_padded[b, ic, h_start:h_end, w_start:w_end]
                            
                            # Element-wise multiply and sum
                            output[b, oc, i, j] += np.sum(window * kernel)
                
                # Add bias
                output[b, oc] += self.bias[oc]
        
        self.cache['output'] = output
        return output
    
    def backward(self, grad_output):
        X = self.cache['X']
        batch_size, in_channels, in_h, in_w = X.shape
        
        # Get output dimensions
        _, _, out_h, out_w = grad_output.shape
        
        # Gradient w.r.t kernels and bias
        dK = np.zeros_like(self.kernels)
        db = np.zeros(self.out_channels)
        
        # Pad input for convolution with grad_output
        if self.padding > 0:
            X_padded = np.pad(
                X, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            X_padded = X
        
        # 1. Compute dK (gradient w.r.t kernels)
        for oc in range(self.out_channels):
            for ic in range(in_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_h
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_w
                        
                        # Window of input
                        window = X_padded[:, ic, h_start:h_end, w_start:w_end]
                        
                        # Gradient w.r.t kernel = input window * grad_output
                        dK[oc, ic] += np.sum(
                            window * grad_output[:, oc, i, j][:, None, None],
                            axis=0
                        )
        
        # 2. Compute db (gradient w.r.t bias)
        db = np.sum(grad_output, axis=(0, 2, 3))
        
        # 3. Compute dX (gradient w.r.t input) - MOST COMPLEX PART!
        # We need to backpropagate through the convolution operation
        
        # Create gradient for padded input
        dX_padded = np.zeros_like(X_padded)
        
        # Rotate kernels 180 degrees for convolution
        rotated_kernels = np.rot90(self.kernels, 2, axes=(2, 3))
        
        # For each output position
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(in_channels):
                    # Get kernel
                    kernel = rotated_kernels[oc, ic]
                    
                    # For each position in grad_output
                    for i in range(out_h):
                        for j in range(out_w):
                            # Add gradient to input at corresponding positions
                            h_start = i * self.stride
                            h_end = h_start + self.kernel_h
                            w_start = j * self.stride
                            w_end = w_start + self.kernel_w
                            
                            # Gradient flows to input = kernel * grad_output
                            dX_padded[b, ic, h_start:h_end, w_start:w_end] += \
                                kernel * grad_output[b, oc, i, j]
        
        # Remove padding
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX, dK, db