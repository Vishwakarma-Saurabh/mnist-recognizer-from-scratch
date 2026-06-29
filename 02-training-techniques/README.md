# 🚀 Project 2: Enhanced MNIST Recognizer with Modern Training Techniques

> Building on Project 1, this implements professional training practices from scratch!

## 🎯 Overview

This project extends the basic neural network from Project 1 with modern deep learning techniques. All implementations are from scratch using only NumPy!

### What's New?

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Adam Optimizer** | Adaptive moment estimation | 5-10x faster convergence |
| **Batch Normalization** | Normalize layer outputs | Higher learning rates, stable training |
| **Multiple Hidden Layers** | Deeper architecture | More expressive power |
| **MixUp Augmentation** | Mix training samples | Better generalization |
| **Label Smoothing** | Soften one-hot labels | Calibrated predictions |
| **LR Scheduling** | Adaptive learning rates | Optimal convergence |
| **Early Stopping** | Stop when no improvement | Save time, prevent overfitting |
| **Advanced Dropout** | Spatial dropout variants | Better regularization |
| **Checkpointing** | Save best model | Resume training, best model |


📚 Key Learnings:- 

✅ Why Adam converges faster than SGD
✅ How Batch Normalization enables higher learning rates
✅ Why MixUp improves generalization
✅ How learning rate scheduling works
✅ When to use different regularization techniques
✅ How to debug training issues systematically
✅ Why model checkpointing is important

## 📁 Project Structure

├── 02-training-techniques/
│   ├── requirements.txt           ✅
│   ├── config.py                  ✅
│   ├── main.py                    ✅
│   ├── README.md                  ✅
│   ├── models/
│   │   ├── __init__.py            ✅
│   │   ├── base_model.py          ✅
│   │   └── layers.py              ✅
│   ├── optimizers/
│   │   ├── __init__.py            ✅
│   │   ├── sgd.py                 ✅
│   │   └── adam.py                ✅
│   ├── schedulers/
│   │   ├── __init__.py            ✅
│   │   └── lr_schedulers.py       ✅
│   ├── data/
│   │   ├── __init__.py            ✅
│   │   ├── dataset.py             ✅
│   │   └── augmentations.py       ✅
│   ├── training/
│   │   ├── __init__.py            ✅
│   │   └── trainer.py             ✅
│   └── utils/
│       ├── __init__.py            ✅
│       └── visualization.py       ✅



## 🚀 Quick Start

# 1. Clone and navigate
git clone https://github.com/Vishwakarma-Saurabh/02-training-techniques.git
cd 02-training-techniques

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python main.py