import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
    
    def step(self, params, grads):
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            
            grad = grads.get(key, 0) + self.weight_decay * params[key]
            
            self.velocities[key] = self.momentum * self.velocities[key] - self.lr * grad
            params[key] += self.velocities[key]
        
        return params