import numpy as np
from scipy.ndimage import rotate, shift, zoom

class DataAugmentation:
    """Data augmentation pipeline"""
    def __init__(self, rotation_range=10, shift_range=0.1, zoom_range=0.1):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
    
    def __call__(self, X, y=None):
        """Apply augmentations to batch"""
        augmented_X = []
        
        for img in X:
            # Reshape to 2D for augmentation
            img_2d = img.reshape(28, 28)
            
            # Random rotation
            if self.rotation_range > 0:
                angle = np.random.uniform(-self.rotation_range, self.rotation_range)
                img_2d = rotate(img_2d, angle, reshape=False, order=1)
            
            # Random shift
            if self.shift_range > 0:
                shift_x = np.random.uniform(-self.shift_range, self.shift_range) * 28
                shift_y = np.random.uniform(-self.shift_range, self.shift_range) * 28
                img_2d = shift(img_2d, [shift_y, shift_x], order=1)
            
            # Random zoom
            if self.zoom_range > 0:
                zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
                zoomed = zoom(img_2d, zoom_factor, order=1)
                
                # Crop back to 28x28
                if zoom_factor > 1:
                    crop = (zoomed.shape[0] - 28) // 2
                    img_2d = zoomed[crop:crop+28, crop:crop+28]
                else:
                    crop = (28 - zoomed.shape[0]) // 2
                    img_2d = np.zeros((28, 28))
                    img_2d[crop:crop+zoomed.shape[0], crop:crop+zoomed.shape[1]] = zoomed
            
            # Reshape back to flattened
            augmented_X.append(img_2d.flatten())
        
        return np.array(augmented_X)

class MixUp:
    """MixUp augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, X, y_one_hot):
        """Apply MixUp"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = X.shape[0]
        
        # Randomly pair samples
        indices = np.random.permutation(batch_size)
        
        # Mix inputs
        mixed_X = lam * X + (1 - lam) * X[indices]
        
        # Mix labels
        mixed_y = lam * y_one_hot + (1 - lam) * y_one_hot[indices]
        
        return mixed_X, mixed_y, lam

def one_hot_encode(y, num_classes=10):
    """Convert labels to one-hot vectors"""
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot