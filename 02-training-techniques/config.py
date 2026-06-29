import numpy as np

class Config:
    # ========== Network Architecture ==========
    INPUT_SIZE = 784
    HIDDEN_SIZES = [256, 128]  # Multiple hidden layers!
    OUTPUT_SIZE = 10
    
    # ========== Training Parameters ==========
    EPOCHS = 100
    BATCH_SIZE = 128
    
    # ========== Optimization ==========
    OPTIMIZER = 'adam'  # 'sgd', 'momentum', 'rmsprop', 'adam', 'adamw'
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001  # L2 regularization
    
    # Momentum parameters
    MOMENTUM = 0.9
    BETA1 = 0.9  # For Adam
    BETA2 = 0.999  # For Adam
    EPSILON = 1e-8
    
    # ========== Learning Rate Scheduling ==========
    LR_SCHEDULE = 'cosine'  # 'step', 'cosine', 'plateau', 'warmup_cosine'
    STEP_SIZE = 30  # For step decay
    GAMMA = 0.5  # For step decay
    WARMUP_EPOCHS = 5  # For warmup
    
    # ========== Regularization ==========
    DROPOUT_RATE = 0.3
    USE_BATCH_NORM = True
    LABEL_SMOOTHING = 0.1
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    
    # ========== Data Augmentation ==========
    AUGMENTATION = True
    ROTATION_RANGE = 10  # degrees
    SHIFT_RANGE = 0.1  # fraction of image size
    ZOOM_RANGE = 0.1
    
    # ========== Training Utilities ==========
    EARLY_STOPPING = True
    PATIENCE = 15  # epochs to wait for improvement
    CHECKPOINT_DIR = 'checkpoints/'
    
    # ========== Logging ==========
    PRINT_EVERY = 5  # epochs
    PLOT_TRAINING = True
    
    @classmethod
    def set_seed(cls, seed=42):
        cls.rng = np.random.default_rng(seed)
        print(f"🔒 Isolated Generator Engine initialized with seed {seed}")