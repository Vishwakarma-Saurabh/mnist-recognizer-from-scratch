import numpy as np

class SGD:
    """Stochastic Gradient Descent with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
        self.t = 0
    
    def step(self, params, grads):
        """Update parameters using SGD with momentum"""
        self.t += 1
        
        for key in params:
            # Safely catch if a matching gradient wasn't calculated for a parameter
            if key not in grads:
                continue
                
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            
            # Incorporate weight decay penalty safely
            grad_with_decay = grads[key] + self.weight_decay * params[key]
            
            # Calculate the modern inertia step
            self.velocities[key] = self.momentum * self.velocities[key] - self.lr * grad_with_decay
            
            # Apply the updated velocity vector to the parameter in-place
            params[key] += self.velocities[key]
        
        return params