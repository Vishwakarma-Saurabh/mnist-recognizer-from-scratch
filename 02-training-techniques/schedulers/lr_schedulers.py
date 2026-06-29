import numpy as np

class LRScheduler:
    """Base learning rate scheduler"""
    def __init__(self, base_lr):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.epoch = 0
    
    def step(self):
        """Update learning rate"""
        self.epoch += 1
        self.current_lr = self.get_lr()
        return self.current_lr
    
    def get_lr(self):
        raise NotImplementedError

class StepLR(LRScheduler):
    """Step learning rate scheduler"""
    def __init__(self, base_lr, step_size=30, gamma=0.5):
        super().__init__(base_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.epoch // self.step_size))

class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler"""
    def __init__(self, base_lr, T_max=100, eta_min=1e-6):
        super().__init__(base_lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * \
               (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2

class WarmupCosineLR(LRScheduler):
    """Warmup + Cosine annealing scheduler"""
    def __init__(self, base_lr, warmup_epochs=5, T_max=100, eta_min=1e-6):
        super().__init__(base_lr)
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self):
        if self.epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (self.epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            return self.eta_min + (self.base_lr - self.eta_min) * \
                   (1 + np.cos(np.pi * progress)) / 2

class ReduceLROnPlateau:
    """Reduce learning rate when metric stops improving"""
    def __init__(self, base_lr, factor=0.5, patience=10, threshold=1e-4):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
        self.epoch = 0
class ReduceLROnPlateau:
    """Reduce learning rate when metric stops improving"""
    def __init__(self, base_lr, mode='min', factor=0.5, patience=10, threshold=1e-4):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
        self.epoch = 0
        
        # Determine performance direction based on mode selection
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be either 'min' (for loss) or 'max' (for accuracy)")
    
    def step(self, metric):
        self.epoch += 1
        
        if self.best_metric is None:
            self.best_metric = metric
            return self.current_lr
        
        # FIXED: Dynamic evaluation logic according to metric target type
        is_improved = False
        if self.mode == 'min':
            # For loss: check if metric is significantly LOWER than best
            if metric < self.best_metric * (1 - self.threshold):
                is_improved = True
        else:
            # For accuracy: check if metric is significantly HIGHER than best
            if metric > self.best_metric * (1 + self.threshold):
                is_improved = True
                
        if is_improved:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.current_lr *= self.factor
                self.wait = 0
                print(f"📉 Learning rate reduced via plateau to {self.current_lr:.6f}")
        
        return self.current_lr
    
import inspect

def get_scheduler(name, learning_rate, T_max, warmup_epochs=0, **kwargs):
    if not name:
        return None
        
    name_lower = name.lower()
    
    # 1. Map string handles to actual Class objects in your module
    # (Adjust these names if your classes are named slightly differently)
    if name_lower == 'cosine':
        scheduler_class = CosineAnnealingLR
    elif name_lower in ['warmup_cosine', 'cosine_warmup']:
        scheduler_class = WarmupCosineLR
    elif name_lower == 'step':
        scheduler_class = StepLR
    elif name_lower == 'plateau':
        scheduler_class = ReduceLROnPlateau
    else:
        raise ValueError(f"Unknown scheduler type: {name}")

    # 2. Collect all available configurations
    available_args = {
        'learning_rate': learning_rate,
        'initial_lr': learning_rate,
        'base_lr': learning_rate,
        'lr': learning_rate,
        'T_max': T_max,
        'warmup_epochs': warmup_epochs,
        **kwargs
    }
    
    # 3. Dynamically scan what the specific class constructor accepts
    sig = inspect.signature(scheduler_class.__init__)
    valid_params = sig.parameters.keys()
    
    # 4. Build a clean dictionary containing ONLY accepted parameters
    filtered_kwargs = {
        k: v for k, v in available_args.items() 
        if k in valid_params
    }
    
    # 5. Instantiate the class safely
    return scheduler_class(**filtered_kwargs)