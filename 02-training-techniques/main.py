import numpy as np
import sys
from config import Config
from models.base_model import NeuralNetwork
from data.dataset import DataLoader
from training.trainer import Trainer
from utils.visualization import Visualizer

def main():
    # 1. Initialize Deterministic Runtime State Environment
    print("⚙️ Initializing global runtime environment...")
    Config.set_seed()
    
    # Extract or fallback to a secure localized NumPy random state engine
    rng_engine = getattr(Config, 'rng', np.random.default_rng(42))
    
    try:
        # 2. Extract Data Matrix Arrays
        print("\n📂 Loading MNIST dataset...")
        data_loader = DataLoader()
        # Handles data pipeline tracking splitting out training, validation, and test arrays
        data_loader.load_mnist(test_size=0.2, val_size=0.1)
        
        # 3. Instantiate Network Graph
        print("🧠 Constructing model graph topology...")
        model = NeuralNetwork(
            input_size=Config.INPUT_SIZE,
            hidden_sizes=Config.HIDDEN_SIZES,
            output_size=Config.OUTPUT_SIZE,
            dropout_rate=Config.DROPOUT_RATE,
            use_batch_norm=Config.USE_BATCH_NORM,
            rng=rng_engine  # FIX: Links reproducibility engine directly to structural parameters
        )
        
        # 4. Generate & Display Advanced Model Summary
        print("\n" + "="*45 + "\n📊 MODEL STRUCTURAL MATRIX CONFIGURATION\n" + "="*45)
        print(f" 🔹 Dimension Mapping  : Input [{Config.INPUT_SIZE}] ➔ Hidden {Config.HIDDEN_SIZES} ➔ Output [{Config.OUTPUT_SIZE}]")
        print(f" 🔹 Regularization     : Dropout Rate: {Config.DROPOUT_RATE} | Batch Normalization: {Config.USE_BATCH_NORM}")
        
        # Display individual layered breakdown statistics
        total_params = 0
        print("\n 🔍 Trainable Parameters Breakdown:")
        for name, tensor in model.params.items():
            param_count = np.prod(tensor.shape)
            total_params += param_count
            print(f"   ▫️ Parameter Matrix Array [{name:10}] : Dimensions {str(tensor.shape):15} | Size: {param_count:,} weights")
            
        print("-"*45)
        print(f" 🔥 Cumulative System Parameters: {total_params:,}")
        print("="*45 + "\n")
        
        # 5. Initialize Optimization Orchestrator
        trainer = Trainer(model, data_loader, Config)
        
        # 6. Execute Forward/Backward Gradient Training Sequences
        history = trainer.train(
            data_loader.X_train, data_loader.y_train,
            data_loader.X_val, data_loader.y_val
        )
        
        # 7. Final Independent Performance Validation Verification
        print("\n🔒 Executing final test performance evaluation...")
        test_loss, test_acc = trainer.test(data_loader.X_test, data_loader.y_test)
        
        # 8. Render Visual Feedback Loops
        if getattr(Config, 'PLOT_TRAINING', True):
            print("\n📈 Rendering analytics dashboard panels...")
            visualizer = Visualizer()
            
            # Loss and convergence trajectories
            visualizer.plot_training_history(history)
            
            # Dynamic learning rate behavior metrics
            visualizer.plot_lr_schedule(history)
            
            # First layer structural weights spatial grid mapping
            visualizer.plot_weights(model)
            
            # Inference sample tracking plots
            visualizer.show_predictions(
                model, 
                data_loader.X_test, 
                data_loader.y_test,
                num_samples=12
            )
            print("✨ Analytics compilation sequence finished successfully.")

    except KeyError as e:
        print(f"\n❌ Execution aborted due to graph processing error: Missing Dictionary Cache Slot {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal exception captured during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()