# 🧠 HOG + SVM Image Classification

This project implements a **handwritten digit classification system** using the **MNIST dataset**.  
Instead of relying on deep learning, it uses **Histogram of Oriented Gradients (HOG)** for feature extraction and a **Support Vector Machine (SVM)** for classification.  
The goal is to demonstrate how traditional feature engineering techniques can achieve strong image recognition performance in a computationally efficient and interpretable way.

---

## 🎯 Problem Statement

To classify handwritten digits (0–9) using handcrafted gradient-based features (HOG) and an SVM classifier,  
and evaluate whether this classical approach can perform comparably to deep learning models on the MNIST dataset.

---

## 💡 Hypothesis

HOG captures meaningful edge and shape information from images,  
and SVM can effectively learn decision boundaries in this high-dimensional space.  
Together, they can produce **high classification accuracy** with **low computational cost** and **high interpretability**.

---

## ⚙️ Methodology

### 1️⃣ Dataset
- **Dataset Used:** MNIST (70,000 grayscale images, 28×28 pixels each)  
- **Train-Test Split:** 75% training, 25% testing  

### 2️⃣ Feature Extraction
- Extracted **Histogram of Oriented Gradients (HOG)** features for each image.
- HOG converts local gradients into orientation histograms, emphasizing shape and structure.
- Visualized HOG images for all ten digit classes.

### 3️⃣ Model Training
- Used a **Support Vector Machine (SVM)** classifier from `scikit-learn`.
- Trained the model on HOG features of the training set.
- Chosen for its robustness in high-dimensional spaces and strong theoretical generalization.

### 4️⃣ Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC Curves**
- **Confusion Matrix**

### 5️⃣ Visualization
- Displayed HOG feature maps for each digit.
- Plotted ROC-AUC curves for all classes.
- Visualized the confusion matrix for detailed performance analysis.

---

## 🧩 Results

| Metric | Value (Approx.) |
|:-------|:----------------|
| Accuracy | ~95–98% |
| Macro F1-score | High (close to 0.95+) |
| ROC-AUC | Near 1.0 for most classes |

- The model successfully distinguishes digits with minimal confusion.
- ROC curves show strong separation between positive and negative classes.

---

## 📊 Conclusion

This project demonstrates that a **HOG + SVM** pipeline can achieve excellent results for image classification without requiring deep neural networks.  
The approach is **interpretable**, **lightweight**, and **effective**, making it suitable for educational and low-compute applications.

---

## 🧩 Tech Stack

- **Language:** Python 3  
- **Libraries:**  
  - `scikit-learn`  
  - `matplotlib`  
  - `numpy`  
  - `cv2` (OpenCV)  
  - `seaborn`  


# Run the main script
python main.py
