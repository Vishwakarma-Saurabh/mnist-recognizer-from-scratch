"""
Optimizers package for gradient descent variants
"""
from optimizers.sgd import SGD
from optimizers.adam import Adam

def get_optimizer(name, **kwargs):
    """Factory function to create optimizer instances"""
    optimizers = {
        'sgd': SGD,
        'momentum': SGD,
        'adam': Adam,
        'adamw': Adam,
    }
    
    name = name.lower()
    if name in optimizers:
        if name == 'momentum':
            kwargs['momentum'] = kwargs.get('momentum', 0.9)
        return optimizers[name](**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

__all__ = [
    'SGD',
    'Adam',
    'get_optimizer'
]