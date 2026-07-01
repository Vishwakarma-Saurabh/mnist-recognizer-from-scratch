# 🧠 Project 3: Convolutional Neural Network from Scratch

> Building CNNs from the ground up using only NumPy - no frameworks allowed!

## 🎯 What is this?

A complete Convolutional Neural Network (CNN) implementation from scratch to classify handwritten digits. This project builds on Projects 1 & 2, now implementing **convolutional layers, pooling, and feature learning** - all with pure NumPy!

## 📁 Project Structure
03-cnn-from-scratch/
├── config.py
├── main.py
├── layers/
│   ├── __init__.py
│   ├── conv2d.py
│   ├── pooling.py
│   ├── flatten.py
│   ├── dense.py
│   ├── activation.py
│   ├── batchnorm.py
│   └── dropout.py
├── models/
│   ├── __init__.py
│   └── cnn.py
├── optimizers/
│   ├── __init__.py
│   ├── adam.py
│   └── sgd.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── augmentations.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
└── requirements.txt
│   
│__ README.md


🏗️ Architecture

Input (28×28×1)
    ↓
Conv2D (1→32, 3×3, padding=1)
    ↓
ReLU + BatchNorm
    ↓
MaxPool (2×2)
    ↓
Conv2D (32→64, 3×3, padding=1)
    ↓
ReLU + BatchNorm
    ↓
MaxPool (2×2)
    ↓
Flatten (64×7×7 = 3136)
    ↓
Dense (3136→128)
    ↓
ReLU + Dropout (0.5)
    ↓
Dense (128→10)
    ↓
Softmax

🎓 Key Takeaways:- 

Convolution is sliding window multiplication
Pooling reduces size, adds translation invariance
Filters learn to detect features (edges → patterns → objects)
Backprop through convolution is complex but powerful
CNNs have fewer parameters but better accuracy

📚 Compared to Previous Projects
Aspect	Project 1	Project 2	Project 3
Type	Dense NN	Enhanced NN	CNN
Layers	2	2 + BatchNorm	Conv + Pool + Dense
Accuracy	95.1%	95.8%	97.6%
Parameters	101K	235K	~50K
Key Concept	Backprop	Optimization	Convolution

🛠️ Requirements

numpy
matplotlib
scikit-learn
scipy
pandas