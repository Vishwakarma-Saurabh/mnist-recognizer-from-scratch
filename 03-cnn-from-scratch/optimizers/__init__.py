from .adam import Adam
from .sgd import SGD

def get_optimizer(name, **kwargs):
    optimizers = {
        'sgd': SGD,
        'momentum': SGD,
        'adam': Adam,
        'adamw': Adam,
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name](**kwargs)