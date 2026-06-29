import numpy as np

class Adam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.weight_decay = weight_decay
        
        # First and second moment estimates
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, params, grads):
        """Update parameters using Adam"""
        self.t += 1
        
        for key in params:
            if key not in grads:
                continue
                
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Use raw gradient for tracking moments (No weight decay blended here!)
            grad = grads[key]
            
            # 1. Update running moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # 2. Compute bias corrections
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 3. Apply decoupled AdamW weight decay penalty directly to parameters
            if self.weight_decay != 0:
                params[key] -= self.lr * self.weight_decay * params[key]
            
            # 4. Perform main adaptive step update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return params