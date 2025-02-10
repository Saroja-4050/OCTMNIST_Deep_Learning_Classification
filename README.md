# OCTMNIST_Deep_Learning_Classification
Deep learning project for multi-class classification using the OCTMNIST dataset. Implements a custom CNN with dropout, early stopping, k-fold cross-validation, and data augmentation. Includes model training, evaluation, performance analysis, and a detailed report with results, confusion matrix, and ROC curves.
# OCTMNIST Deep Learning Classification

A deep learning project for multi-class classification using the OCTMNIST dataset. This project implements a custom Convolutional Neural Network (CNN) and applies various techniques to improve model accuracy and efficiency.

---

## 📌 **Project Overview**
This project focuses on the classification of Optical Coherence Tomography (OCT) images into multiple classes using a deep learning approach. The key objectives include:
- Building a **custom CNN model** for image classification.
- Implementing **techniques** like dropout, early stopping, k-fold cross-validation, and data augmentation.
- Evaluating the model using **accuracy, loss curves, confusion matrices, and ROC curves**.
- **Improving accuracy** by applying learning rate schedulers and hyperparameter tuning.

---

## 📂 **Dataset**
- **Dataset Name**: OCTMNIST
- **Number of Samples**: 223,995
- **Classes**: 4 (Imbalance handled with SMOTE)
- **Image Size**: 28x28 grayscale images
- **Source**: MedMNIST dataset

### 📊 **Dataset Preprocessing**
- Converted grayscale images to normalized pixel values \([0,1]\).
- Addressed **class imbalance** using **SMOTE**.
- Applied **contrast enhancement** (CLAHE) to improve visibility of image features.
- Performed **data augmentation** (flipping, rotation, brightness adjustment).

---

## 🏗 **Model Architecture**
The CNN model consists of:
- **Convolutional Layers**: Extract spatial features
- **Batch Normalization**: Normalize activations
- **Dropout**: Prevent overfitting
- **Fully Connected Layers**: For final classification

---

## 🚀 **Implementation Details**
### 1️⃣ **Training Setup**
- **Optimizer**: Adam (`lr=0.001`)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 10 (Early stopping used)

### 2️⃣ **Performance Metrics**
- **Accuracy**: ~80.19%
- **Precision**: 0.8090
- **Recall**: 0.8019
- **F1 Score**: 0.8037

### 3️⃣ **Model Improvements**
✅ **K-Fold Cross-Validation** (5 Folds)  
✅ **Early Stopping** (Patience: 3)  
✅ **Learning Rate Scheduler** (Reduce on Plateau)  
✅ **Regularization (Dropout, BatchNorm)**  


