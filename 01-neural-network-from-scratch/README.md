# 🧠 MNIST Handwritten Digit Recognizer

> A 2-layer neural network built from scratch using only NumPy

## 🎯 What is this?

This project implements a neural network to recognize handwritten digits (0-9) from the MNIST dataset. **No TensorFlow, PyTorch, or any deep learning frameworks** - just pure NumPy!

## 📁 Project Structure
mnist_recognizer/
├── config.py # Hyperparameters (learning rate, batch size, etc.)
├── neural_network.py # Core NN: forward/backward propagation
├── utils.py # Data loading and visualization
├── train.py # Training loop
├── main.py # Run everything
└── requirements.txt # Dependencies


## 🚀 Quick Start

# 1. Clone the repo
git clone https://github.com/Vishwakarma-Saurabh/mnist-recognizer.git
cd mnist-recognizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python main.py

📊 What you'll see
Loading MNIST dataset...
Training set: 49,000 samples
Test set: 14,000 samples

Network Architecture:
  Input: 784 neurons
  Hidden: 128 neurons  
  Output: 10 neurons

Epoch 1: Train Acc: 86.71%, Val Acc: 86.39%
Epoch 10: Train Acc: 93.60%, Val Acc: 93.39%
Epoch 50: Train Acc: 97.44%, Val Acc: 96.47%

Test Accuracy: 96.46% ✅

🔧 Customize
Edit config.py to change:
HIDDEN_SIZE: Number of hidden neurons
LEARNING_RATE: Learning speed
BATCH_SIZE: Mini-batch size
EPOCHS: Training iterations
DROPOUT_RATE: Dropout probability

🧪 Test your own digit
from neural_network import NeuralNetwork
import numpy as np

# Load trained model
model = NeuralNetwork(784, 128, 10)
# ... load weights ...

# Predict your digit (28x28 grayscale image)
your_digit = np.array(...).reshape(1, -1) / 255.0
prediction = model.predict(your_digit)
print(f"Predicted: {prediction[0]}")

📝 Requirements
Python 3.8+
NumPy
Matplotlib
scikit-learn
pandas (indirecly, required in other dependencies)

🤔 Why from scratch?
Building from scratch helps you understand:

How forward/backward propagation actually works

Why weight initialization matters

What happens during gradient descent

The math behind neural networks

📚 Key Concepts Implemented
✅ Forward propagation with ReLU & Softmax

✅ Backpropagation with chain rule

✅ Mini-batch gradient descent

✅ Xavier weight initialization

✅ L2 regularization

✅ Dropout

✅ Cross-entropy loss

✅ Gradient checking

📈 Results
Dataset	Accuracy
Training	~97%
Validation	~96%
Test	~95%

📄 License
MIT License - feel free to use and modify!
