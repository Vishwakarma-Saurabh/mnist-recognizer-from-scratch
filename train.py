import numpy as np
from config import Config

class Trainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, X_train, y_train, X_val, y_val):
        n_samples = X_train.shape[0]
        
        # Convert labels to one-hot
        y_train_one_hot = self.data_loader.one_hot_encode(y_train)
        
        # Shuffle and create mini-batches
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_one_hot[indices]
        
        total_batches = int(np.ceil(n_samples / Config.BATCH_SIZE))
        
        for batch_idx in range(total_batches):
            start = batch_idx * Config.BATCH_SIZE
            end = min(start + Config.BATCH_SIZE, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass
            y_pred = self.model.forward(X_batch, training=True, 
                                      dropout_rate=Config.DROPOUT_RATE)
            
            # Backward pass
            gradients = self.model.backward(y_batch, l2_lambda=Config.L2_LAMBDA)
            
            # Update weights
            self.model.update_params(gradients, learning_rate=Config.LEARNING_RATE)
        
        # Evaluate after epoch
        y_pred_train = self.model.forward(X_train, training=False)
        y_pred_val = self.model.forward(X_val, training=False)
        
        train_loss = self.model.compute_loss(
            self.data_loader.one_hot_encode(y_train), 
            y_pred_train, 
            l2_lambda=Config.L2_LAMBDA
        )
        val_loss = self.model.compute_loss(
            self.data_loader.one_hot_encode(y_val), 
            y_pred_val, 
            l2_lambda=Config.L2_LAMBDA
        )
        train_acc = self.model.accuracy(X_train, y_train)
        val_acc = self.model.accuracy(X_val, y_val)
        
        return train_loss, val_loss, train_acc, val_acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        print("Starting training...")
        print("-" * 60)
        
        for epoch in range(1, epochs + 1):
            train_loss, val_loss, train_acc, val_acc = self.train_epoch(
                X_train, y_train, X_val, y_val
            )
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Acc: {val_acc:.2f}%")
        
        print("-" * 60)
        print("Training complete!")
        
        return self.history