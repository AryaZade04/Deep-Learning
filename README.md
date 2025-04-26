# 🧠 Deep Learning Project: Handwritten Digit Recognition & Heart Disease Prediction

This repository contains a deep learning notebook (`AIML_CA4.ipynb`) featuring two real-world scenarios implemented using TensorFlow and Keras:

1. ✍️ **Scenario 1**: Handwritten Digit Recognition using MNIST Dataset
2. ❤️ **Scenario 2**: Heart Disease Prediction using Patient Medical Records

---


---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- Google Colab

---

## ✍️ Scenario 1: Handwritten Digit Recognition using MNIST Dataset

### 📝 Introduction

A startup is building a mobile app that reads handwritten receipts and logs expenses. It needs a system to recognize digits (0–9) from scanned images. We implement a **Feedforward Neural Network (FNN)** to solve this using the MNIST dataset.

### 🎯 Objective

- Classify handwritten digits from 28×28 grayscale images using FNN
- Achieve high accuracy with minimal computation
- Implement using TensorFlow/Keras

### 📊 Dataset

- **MNIST** benchmark dataset
- 60,000 training images + 10,000 test images
- Each image is 28×28 pixels, grayscale
- Labels: digits from 0 to 9

### 🧩 Model Architecture

- **Flatten** layer to convert image to 1D vector
- **Dense layers** with ReLU activation (128, then 64 neurons)
- **Output layer**: Dense with 10 units + softmax activation

### ⚙️ Training Details

- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Batch size: 32
- Epochs: 10
- Validation split: 20%

### 🧪 Results

- ✅ **Test Accuracy**: ~97.6%
- Training and validation curves show high performance and good generalization

### 📈 Future Improvements

- Use **CNNs** for better spatial understanding
- Apply **dropout** and **batch normalization**
- Try **data augmentation** for generalization
- Tune hyperparameters and architecture

---

## ❤️ Scenario 2: Heart Disease Prediction using ANN

### 📝 Introduction

Heart disease is a global concern. Early detection via patient medical data (e.g., age, cholesterol, blood pressure) can help in timely diagnosis. This scenario implements an ANN to predict whether a patient is at risk.

### 🎯 Objective

- Build a **binary classification** model using ANN
- Use Keras to build and train the network
- Handle **class imbalance** using computed class weights
- Evaluate using accuracy and classification report

### 📊 Dataset

- Medical records (CSV) with features like age, blood pressure, cholesterol
- Target variable: `target`  
  - 1 → has heart disease  
  - 0 → does not have heart disease

### 🧩 Model Architecture

- **Input**: Standardized numerical features
- **Hidden layers**: Dense (64, 32 neurons) with ReLU and Dropout
- **Output layer**: Dense(1) with **sigmoid** activation
- Loss: `binary_crossentropy`
- Optimizer: `Adam`

### ⚖️ Class Imbalance Handling

- Used `compute_class_weight()` from sklearn to prevent bias toward the majority class

### 🧪 Evaluation

- ✅ **Test Accuracy**: ~88%
- Accuracy/loss plots for training and validation
- Confusion matrix and classification report

### 🩺 Sample Output

```plaintext
Test Accuracy: 88.25%

Classification Report:
              precision    recall  f1-score   support
           0       0.89      0.90      0.89       102
           1       0.87      0.86      0.86        78
