# config.py
import numpy as np

class Config:
    # Network Architecture
    INPUT_SIZE = 784      # 28x28 pixels flattened
    HIDDEN_SIZE = 128     # Hidden layer neurons
    OUTPUT_SIZE = 10      # 10 digits (0-9)
    
    # Training Parameters
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64       # Mini-batch size
    EPOCHS = 50
    L2_LAMBDA = 0.001    # Regularization strength
    DROPOUT_RATE = 0.2   # Probability of dropping neurons
    
    # Initialization
    WEIGHT_INIT_SCALE = 0.01
    
    # Data
    TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.1
    RANDOM_SEED = 42

    @staticmethod
    def set_seed():
        np.random.seed(Config.RANDOM_SEED)