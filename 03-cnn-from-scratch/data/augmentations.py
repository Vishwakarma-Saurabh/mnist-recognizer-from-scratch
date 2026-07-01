import numpy as np
from scipy.ndimage import rotate, shift, zoom

class DataAugmentation:
    """Data augmentation for images"""
    def __init__(self, rotation_range=10, shift_range=0.1, zoom_range=0.1, rng=None):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.rng = rng if rng is not None else np.random.default_rng(42)
    
    def __call__(self, X, y=None):
        """Apply augmentations to batch"""
        augmented_X = []
        
        for img in X:
            # img shape: (channels, height, width)
            if len(img.shape) == 3:
                img_2d = img[0]  # Extract single channel
            else:
                img_2d = img
            
            # Random rotation
            if self.rotation_range > 0:
                angle = self.rng.uniform(-self.rotation_range, self.rotation_range)
                img_2d = rotate(img_2d, angle, reshape=False, order=1)
            
            # Random shift
            if self.shift_range > 0:
                shift_x = self.rng.uniform(-self.shift_range, self.shift_range) * 28
                shift_y = self.rng.uniform(-self.shift_range, self.shift_range) * 28
                img_2d = shift(img_2d, [shift_y, shift_x], order=1)
            
            # Random zoom
            if self.zoom_range > 0:
                zoom_factor = self.rng.uniform(1 - self.zoom_range, 1 + self.zoom_range)
                zoomed = zoom(img_2d, zoom_factor, order=1)
                
                if zoom_factor > 1:
                    crop = (zoomed.shape[0] - 28) // 2
                    img_2d = zoomed[crop:crop+28, crop:crop+28]
                else:
                    crop = (28 - zoomed.shape[0]) // 2
                    new_img = np.zeros((28, 28))
                    new_img[crop:crop+zoomed.shape[0], crop:crop+zoomed.shape[1]] = zoomed
                    img_2d = new_img
            
            # Reshape back
            augmented_X.append(img_2d.reshape(1, 28, 28))
        
        return np.array(augmented_X)