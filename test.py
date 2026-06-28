import numpy as np
from config import Config

class Tester:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
    
    def gradient_check(self, X, y, epsilon=1e-7):
        print("Performing gradient check...")
        
        # Convert to one-hot
        y_one_hot = self.data_loader.one_hot_encode(y)
        
        # Forward pass
        self.model.forward(X, training=False)
        
        # Get analytical gradients
        grads = self.model.backward(y_one_hot, l2_lambda=0)
        
        # Check gradients for each parameter
        params = [('W1', self.model.W1, grads['dW1']),
                  ('b1', self.model.b1, grads['db1']),
                  ('W2', self.model.W2, grads['dW2']),
                  ('b2', self.model.b2, grads['db2'])]
        
        for name, param, grad in params:
            # Compute numerical gradient
            num_grad = np.zeros_like(param)
            
            # Iterate over all parameters (simplified - check few for speed)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            count = 0
            max_checks = 10  # Check only first 10 parameters for speed
            
            while not it.finished and count < max_checks:
                idx = it.multi_index
                count += 1
                
                # f(x+h)
                param_copy1 = param.copy()
                param_copy1[idx] += epsilon
                
                # Temporarily set parameter
                original = param[idx]
                param[idx] = param_copy1[idx]
                y_pred_plus = self.model.forward(X, training=False)
                loss_plus = self.model.compute_loss(y_one_hot, y_pred_plus, l2_lambda=0)
                
                # f(x-h)
                param_copy2 = param.copy()
                param_copy2[idx] -= epsilon
                param[idx] = param_copy2[idx]
                y_pred_minus = self.model.forward(X, training=False)
                loss_minus = self.model.compute_loss(y_one_hot, y_pred_minus, l2_lambda=0)
                
                # Numerical gradient
                num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Restore original
                param[idx] = original
                it.iternext()
            
            # Compare gradients
            if np.any(num_grad != 0):
                rel_error = np.abs(grad.flatten()[:max_checks] - num_grad.flatten()[:max_checks]) / \
                           np.maximum(np.abs(grad.flatten()[:max_checks]), 
                                    np.abs(num_grad.flatten()[:max_checks]) + 1e-8)
                max_rel_error = np.max(rel_error)
                print(f"{name}: Max relative error = {max_rel_error:.6f}")
                
                if max_rel_error < 1e-6:
                    print(f"  ✓ Gradient check passed!")
                else:
                    print(f"  ✗ Gradient check failed! Check implementation.")
    
    def test_forward_shape(self, X):
        try:
            output = self.model.forward(X)
            expected_shape = (X.shape[0], Config.OUTPUT_SIZE)
            assert output.shape == expected_shape, \
                f"Output shape {output.shape} != {expected_shape}"
            print(f"✓ Forward pass shape test passed: {output.shape}")
            return True
        except Exception as e:
            print(f"✗ Forward pass shape test failed: {e}")
            return False