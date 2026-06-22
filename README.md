# 🧠 Deep Learning with ANN & CNN
### A Comprehensive Study of Neural Networks for Regression & Classification

**Student:** Muhammad Ibtihaj  
**Registration No:** SP23-BAI-037

> *Exploring Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) across diverse machine learning tasks: regression, MNIST classification, and CIFAR-10 image classification.*

---

## Overview

This project demonstrates the application and comparison of deep learning techniques across three distinct machine learning tasks:

| # | Task | Dataset | Model | Accuracy |
|---|---|---|---|---|
| 1 | 🏠 Regression | California Housing | ANN | R² = 0.4062 |
| 2 | ✍️ Classification | MNIST | ANN | 97.32% |
| 3 | 🖼️ Classification | CIFAR-10 | CNN | 59.55% (Model 4) |

---

## Table of Contents

- [Part 1: California Housing Regression (ANN)](#part-1--california-housing-regression-ann)
- [Part 2: MNIST Classification (ANN)](#part-2--mnist-classification-ann)
- [Part 3: CIFAR-10 Classification (CNN)](#part-3--cifar-10-classification-cnn)
- [Installation & Setup](#installation--setup)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)

---

## Part 1 — 🏠 Regression on California Housing Dataset (ANN)

### Dataset
The **California Housing** dataset contains information on housing districts across California. This is a classic regression problem used to predict median house values.

- **Samples:** 20,640
- **Features:** 8 (median income, house age, average rooms, etc.)
- **Target:** Median house value (continuous)
- **Train/Test Split:** 80/20

### Model Architecture

```
Input Layer     →  8 features
Dense Layer 1   →  32 units, ReLU activation
Dense Layer 2   →  16 units, ReLU activation
Output Layer    →  1 unit (linear activation - regression)
```

### Preprocessing
- **Feature Scaling:** StandardScaler normalization
- **Dataset Class:** Custom PyTorch Dataset with on-the-fly scaling
- **Batch Size:** 32
- **Optimization:** SGD with learning rate = 0.001, weight decay = 1e-4

### Training Details
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Epochs:** 100
- **Validation Strategy:** 20% validation split

### Results

| Metric | Value |
|---|---|
| **Mean Squared Error (MSE)** | 0.7781 |
| **Mean Absolute Error (MAE)** | 0.6561 |
| **R² Score** | 0.4062 |

### Training Progress

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 10 | 1.3366 | 1.3104 |
| 20 | 1.3355 | 1.3091 |
| 30 | 1.3049 | 1.2704 |
| 40 | 0.5850 | 0.5704 |
| 50 | 0.4726 | 0.4520 |
| 60 | 0.4485 | 0.4447 |
| 70 | 0.4311 | 0.4687 |
| 80 | 0.4163 | 0.5289 |
| 90 | 0.4020 | 0.6264 |
| 100 | 0.3890 | 0.7781 |

**Observations:**
- Significant loss reduction from epoch 30-50 indicates effective learning
- Validation loss increases slightly after epoch 60, suggesting potential overfitting
- Final R² of 0.4062 indicates the model explains ~41% of variance in house prices

---

## Part 2 — ✍️ Classification on MNIST Dataset (ANN)

### Dataset
The **MNIST** dataset is a benchmark dataset of handwritten digits, one of the most popular datasets in machine learning.

- **Samples:** 70,000 (60,000 train / 10,000 test)
- **Image Size:** 28×28 pixels (grayscale)
- **Classes:** 10 (digits 0–9)
- **Input Flattened Size:** 784 features

### Model Architecture

```
Input Layer     →  784 units (28×28 flattened)
Dense Layer 1   →  128 units, ReLU activation
Dense Layer 2   →  64 units, ReLU activation
Output Layer    →  10 units, Softmax activation (classification)
```

### Preprocessing
- **Normalization:** Pixel values normalized to [-1, 1] using (image - 0.5) / 0.5
- **Label Encoding:** One-hot encoding
- **Device:** GPU acceleration (if available, else CPU)
- **Batch Size:** 32

### Training Details
- **Loss Function:** Cross Entropy Loss
- **Optimizer:** SGD with learning rate = 0.01, weight decay = 1e-4
- **Epochs:** 20
- **Validation:** Test set evaluation

### Results

| Metric | Value |
|---|---|
| **Accuracy** | 97.32% |
| **Precision (Weighted)** | 97.34% |
| **Recall (Weighted)** | 97.32% |
| **F1 Score (Weighted)** | 97.32% |

### Training Progress

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 5 | 0.2766 | 0.2437 |
| 10 | 0.1347 | 0.1321 |
| 15 | 0.0893 | 0.1016 |
| 20 | 0.0650 | 0.0876 |

**Observations:**
- Exceptional performance with 97.32% accuracy
- Rapid convergence with steep loss reduction in first 10 epochs
- Excellent precision, recall, and F1 scores indicate balanced classification
- Minimal overfitting despite reaching very low training loss

---

## Part 3 — 🖼️ Classification on CIFAR-10 Dataset (CNN)

### Dataset
The **CIFAR-10** dataset consists of 60,000 32×32 color images across 10 object categories, significantly more challenging than MNIST.

- **Samples:** 60,000 (50,000 train / 10,000 test)
- **Image Size:** 32×32 pixels (RGB - 3 channels)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Train/Val Split:** 80/20

### Preprocessing
- **Normalization:** Pixel values normalized to [0, 1] (divide by 255)
- **Label Encoding:** One-hot encoding (10 classes)
- **Batch Size:** 32
- **Validation Split:** 20%

### Model Architectures

All models use **Adam optimizer** (learning rate = 0.001) and **Categorical Cross-Entropy loss**.

#### **Model 1: Shallow CNN (2 Conv Layers)**

```
Conv2D          →  2 filters, 3×3, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Conv2D          →  4 filters, 3×3, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Flatten         →  256 units
Dense           →  128 units, ReLU
Output          →  10 units, Softmax
```

**Total Parameters:** 34,318  
**Test Accuracy:** 54.57%  
**Test Loss:** 1.3053

---

#### **Model 2: Medium CNN (3 Conv Layers)**

```
Conv2D          →  4 filters, 3×3, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Conv2D          →  8 filters, 3×3, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Conv2D          →  16 filters, 3×3, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Flatten         →  256 units
Dense           →  128 units, ReLU
Output          →  10 units, Softmax
```

**Total Parameters:** 35,762  
**Test Accuracy:** 59.01%  
**Test Loss:** 1.1689

---

#### **Model 3: Single Conv Layer (Shallow)**

```
Conv2D          →  4 filters, 3×3, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Flatten         →  1024 units
Dense           →  128 units, ReLU
Output          →  10 units, Softmax
```

**Total Parameters:** 132,602 (more dense layer params)  
**Test Accuracy:** 55.55%  
**Test Loss:** 1.2806

---

#### **Model 4: 5×5 Kernel CNN (Best Performance) ⭐**

```
Conv2D          →  4 filters, 5×5, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Conv2D          →  8 filters, 5×5, ReLU, padding='same'
MaxPooling2D    →  2×2 pool
Flatten         →  512 units
Dense           →  128 units, ReLU
Output          →  10 units, Softmax
```

**Total Parameters:** 68,066  
**Test Accuracy:** 59.55% (Best)  
**Test Loss:** 1.1774

---

### CIFAR-10 Results Comparison

| Model | Architecture | Parameters | Test Accuracy | Test Loss | Remarks |
|---|---|---|---|---|---|
| **Model 1** | 2 Conv Layers (3×3) | 34,318 | 54.57% | 1.3053 | Underfitting - too simple |
| **Model 2** | 3 Conv Layers (3×3) | 35,762 | 59.01% | 1.1689 | Good performance |
| **Model 3** | 1 Conv Layer (3×3) | 132,602 | 55.55% | 1.2806 | High dense params lead to overfitting |
| **Model 4** | 2 Conv Layers (5×5) | 68,066 | **59.55%** | 1.1774 | **Best overall - larger receptive field** |

### Key Insights for CIFAR-10

- 🔷 **Larger Kernels Win:** Model 4 with 5×5 kernels outperformed 3×3 kernels, capturing larger spatial patterns
- 🔷 **Depth vs. Width Trade-off:** Model 2 (3 layers) performed well, but Model 4 (2 layers, larger kernels) was superior
- 🔷 **Dense Layer Bloat:** Model 3's large fully connected layer (132k params) led to overfitting despite single conv layer
- 🔷 **Optimal Balance:** Model 4 achieved the best balance with moderate depth and appropriate kernel size

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch (for Parts 1 & 2)
- TensorFlow/Keras (for Part 3)
- scikit-learn
- NumPy, Matplotlib, Seaborn

### Install Dependencies

```bash
# Install PyTorch (CPU or GPU)
pip install torch torchvision

# Install TensorFlow & Keras
pip install tensorflow

# Install other requirements
pip install scikit-learn numpy matplotlib seaborn
```

### Run Each Part

```bash
# Part 1 — California Housing Regression (PyTorch ANN)
python part1_california_housing_regression.py

# Part 2 — MNIST Classification (PyTorch ANN)
python part2_mnist_classification.py

# Part 3 — CIFAR-10 Classification (Keras CNN)
python part3_cifar10_classification.py
```

---

## Results Summary

### 📊 Performance Comparison

| Task | Model Type | Accuracy/R² | Key Metric |
|---|---|---|---|
| **Regression** | ANN (2 hidden layers) | R² = 0.4062 | MAE: 0.6561 |
| **MNIST** | ANN (2 hidden layers) | 97.32% | Excellent convergence |
| **CIFAR-10** | CNN (Multiple variants) | 59.55% | Limited by dataset complexity |

---

## Key Findings

### 1. **ANN on Tabular Data (California Housing)**
   - ✅ ANNs are effective for structured regression problems
   - ✅ Quick convergence with proper preprocessing
   - ⚠️ R² of 0.40 suggests room for improvement (ensemble methods, feature engineering needed)

### 2. **ANN on Simple Image Data (MNIST)**
   - ✅ **Exceptional performance** with 97.32% accuracy
   - ✅ Demonstrates ANN's capability for image classification on simple datasets
   - ✅ Small image size (28×28) and simple patterns suit fully connected networks well
   - ✅ Fast training and inference

### 3. **CNN on Complex Image Data (CIFAR-10)**
   - ✅ CNNs leverage spatial structure through convolutional layers
   - ⚠️ Best accuracy of 59.55% indicates CIFAR-10's complexity
   - 📌 **5×5 kernels > 3×3 kernels** for capturing larger spatial patterns
   - 📌 Architectural choices matter significantly (kernel size > depth in this case)
   - 💡 Suggests need for: Data augmentation, batch normalization, dropout, or deeper architectures

### 4. **ANN vs. CNN**
   - **MNIST:** ANN wins (97.32%) due to small, simple images
   - **CIFAR-10:** CNN better suited, but still limited without advanced techniques
   - **Conclusion:** Task complexity and data characteristics determine optimal architecture

---

## Architecture Design Lessons

| Aspect | Finding |
|---|---|
| **Kernel Size** | Larger kernels (5×5) > smaller kernels (3×3) for CIFAR-10 |
| **Network Depth** | 2-3 conv layers optimal; deeper layers don't guarantee better performance |
| **Dense Layers** | Too many dense layer parameters lead to overfitting |
| **Preprocessing** | Feature scaling critical for ANN; normalization essential for CNN |
| **Simple vs. Complex Data** | ANN sufficient for MNIST; CNN needed for CIFAR-10 |

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-yellowgreen?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?style=flat-square)

---

## Conclusion

This project successfully demonstrates:

1. **ANN Effectiveness** for both regression and simple image classification tasks
2. **CNN Superiority** for complex image datasets through spatial feature extraction
3. **Architectural Importance** - careful design choices significantly impact performance
4. **Task-Specific Approaches** - different problems require different neural network designs

The progression from simple tabular regression → simple image classification → complex image classification illustrates how neural network complexity must scale with task difficulty.

---

**Author:** Muhammad Ibtihaj (SP23-BAI-037)
