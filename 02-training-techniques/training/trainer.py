import numpy as np
import time
import os
import pickle
from data.augmentations import DataAugmentation, MixUp, one_hot_encode
from optimizers import get_optimizer
from schedulers.lr_schedulers import get_scheduler

class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # Pull isolated generator from the network architecture
        self.generator = getattr(model, 'rng', None) or np.random.default_rng(42)
        
        # Initialize optimization & scheduling
        self.optimizer = get_optimizer(
            config.OPTIMIZER,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = get_scheduler(
            config.LR_SCHEDULE,
            config.LEARNING_RATE,
            T_max=config.EPOCHS,
            warmup_epochs=config.WARMUP_EPOCHS
        )
        
        # Initialize modern augmentations
        self.augmenter = DataAugmentation(
            rotation_range=config.ROTATION_RANGE,
            shift_range=config.SHIFT_RANGE,
            zoom_range=config.ZOOM_RANGE
        )
        self.mixup = MixUp(config.MIXUP_ALPHA) if config.USE_MIXUP else None
        
        # Performance & telemetry history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': [], 'time': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_params = None
        self.patience_counter = 0
        
        if getattr(config, 'CHECKPOINT_DIR', None):
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            
    def _evaluate_batched(self, X, y, batch_size=256):
        """Evaluates loss and accuracy in safe batches to protect cache memory."""
        n_samples = X.shape[0]
        total_loss = 0.0
        correct_predictions = 0
        total_batches = int(np.ceil(n_samples / batch_size))
        
        y_one_hot = one_hot_encode(y)
        
        for b in range(total_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            y_batch_oh = y_one_hot[start:end]
            
            # Forward evaluation mode (freezes BatchNorm/Dropout cleanly)
            y_pred = self.model.forward(X_batch, training=False)
            
            # Compute raw batch loss
            total_loss += self.model.compute_loss(
                y_batch_oh, y_pred, 
                l2_lambda=0,  # L2 regularization will be added globally below
                label_smoothing=0
            ) * (end - start)
            
            # Track correct classification counts
            pred_digits = np.argmax(y_pred, axis=1)
            correct_predictions += np.sum(pred_digits == y_batch)
            
        # Add the structural L2 penalty once globally for the entire set evaluation
        l2_penalty = 0.0
        if self.config.WEIGHT_DECAY > 0:
            for layer in self.model.layers:
                if hasattr(layer, 'W'):
                    l2_penalty += 0.5 * self.config.WEIGHT_DECAY * np.sum(layer.W ** 2)
                    
        avg_loss = (total_loss / n_samples) + (l2_penalty / n_samples)
        accuracy_percentage = (correct_predictions / n_samples) * 100.0
        
        return avg_loss, accuracy_percentage

    def train_epoch(self, X_train, y_train, X_val, y_val):
        """Runs a complete forward/backward optimization loop safely over the dataset."""
        n_samples = X_train.shape[0]
        batch_size = self.config.BATCH_SIZE
        total_batches = int(np.ceil(n_samples / batch_size))
        
        y_train_one_hot = one_hot_encode(y_train)
        
        # Permute using our deterministic localized seed generator
        indices = self.generator.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_one_hot[indices]
        
        running_train_loss = 0.0
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Apply Augmentation and MixUp pipeline
            if getattr(self.config, 'AUGMENTATION', False):
                X_batch = self.augmenter(X_batch)
            if self.mixup and getattr(self.config, 'USE_MIXUP', False):
                X_batch, y_batch, _ = self.mixup(X_batch, y_batch)
                
            # Core Backpropagation execution
            y_pred = self.model.forward(X_batch, training=True)
            
            # Accumulate metrics online during the training pass
            batch_loss = self.model.compute_loss(
                y_batch, y_pred, 
                l2_lambda=self.config.WEIGHT_DECAY, 
                label_smoothing=self.config.LABEL_SMOOTHING
            )
            running_train_loss += batch_loss * (end - start)
            
            # Backprop pass (passes lambda=0 if optimizer implements decoupled AdamW weight decay)
            self.model.backward(y_batch, l2_lambda=self.config.WEIGHT_DECAY)
            self.model.update_params(self.optimizer)
            
        # Calculate final training loss using our online accumulator
        train_loss = running_train_loss / n_samples
        
        # Efficient training set accuracy run (no heavy loss calculation overhead)
        train_acc = self.model.accuracy(X_train, y_train)
        
        # Secure validation run using batched isolation logic
        val_loss, val_acc = self._evaluate_batched(X_val, y_val, batch_size=batch_size)
        
        return train_loss, val_loss, train_acc, val_acc

    def train(self, X_train, y_train, X_val, y_val):
        """Full training sequence including learning rate schedules and early stopping checkpoints."""
        print("=" * 60)
        print("🚀 DEPLOYING PRODUCTION OPTIMIZED TRAINER")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.EPOCHS + 1):
            epoch_start = time.time()
            
            train_loss, val_loss, train_acc, val_acc = self.train_epoch(
                X_train, y_train, X_val, y_val
            )
            
            # Update learning rates via schedulers
            current_lr = self.config.LEARNING_RATE
            if self.scheduler:
                current_lr = self.scheduler.step(val_acc) if 'Plateau' in type(self.scheduler).__name__ else self.scheduler.step()
                
            epoch_time = time.time() - epoch_start
            
            # Telemetry compilation
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['time'].append(epoch_time)
            
            # Checkpoint performance evaluation
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_params = self.model.get_parameters()
                self.patience_counter = 0
                
                if getattr(self.config, 'CHECKPOINT_DIR', None):
                    checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pkl')
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'epoch': epoch, 'model_params': self.best_model_params,
                            'val_acc': val_acc, 'val_loss': val_loss
                        }, f)
            else:
                self.patience_counter += 1
                
            if epoch % self.config.PRINT_EVERY == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{self.config.EPOCHS} | LR: {current_lr:.6f} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
                
            if getattr(self.config, 'EARLY_STOPPING', False) and self.patience_counter >= self.config.PATIENCE:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch}. Best Val Acc: {self.best_val_acc:.2f}%")
                break
                
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"✅ Optimization complete in {total_time/60:.2f} minutes.")
        print("=" * 60)
        
        if self.best_model_params:
            self.model.set_parameters(self.best_model_params)
            
        return self.history

    def test(self, X_test, y_test):
        """Final clean performance validation evaluation loop."""
        test_loss, test_acc = self._evaluate_batched(X_test, y_test, batch_size=self.config.BATCH_SIZE)
        print(f"\n📊 Final Evaluated Test Results:\n Loss: {test_loss:.4f}\n Accuracy: {test_acc:.2f}%")
        return test_loss, test_acc