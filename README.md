# 🍎🍌 Apple vs Banana Image Classification – Classical Computer Vision Mini Project

This project demonstrates a structured Computer Vision pipeline using traditional machine learning techniques and feature engineering.

---

## 📌 Overview

The objective of this project is to classify images of apples and bananas using:

- Image preprocessing
- Raw pixel features
- HOG (Histogram of Oriented Gradients)
- Logistic Regression classifier

The project also evaluates robustness under controlled perturbations such as Gaussian noise and brightness shifts.

---

## 🧠 Computer Vision Pipeline

1. Load raw images  
2. Resize with aspect ratio preservation  
3. Pad images to 128×128  
4. Convert to grayscale  
5. Normalize pixel values (0–1)  
6. Feature Extraction:
   - Raw Pixel Flattening (16384 features)
   - HOG Features (8100 features)
7. Train Logistic Regression model  
8. Evaluate using Accuracy and Confusion Matrix  

---

## 🛠 Tools & Libraries

- Python  
- OpenCV  
- NumPy  
- Scikit-learn  
- Scikit-image  

---

## 📊 Dataset

- Apple images: 310  
- Banana images: 310  
- Clean background and controlled lighting  
- Dataset not included in repository (.gitignore)

---

# 📈 Week 3 – Feature Engineering Comparison

## ✅ Baseline Results (Clean Data)

| Method        | Feature Length | Accuracy | Training Time |
|--------------|---------------|----------|---------------|
| Raw Pixels   | 16384        | 100%     | ~0.12 sec     |
| HOG Features | 8100         | 100%     | ~0.16 sec     |

Confusion Matrix (Both Methods):
[[63 0]
[ 0 61]]


### 🔎 Observations (Baseline)

- Both Raw Pixel and HOG features achieved perfect accuracy.
- HOG reduced feature dimensionality by ~50% (16384 → 8100).
- Training time for HOG was slightly higher due to feature computation overhead.
- The dataset is linearly separable under both representations due to:
  - Clean background  
  - Controlled lighting  
  - Distinct object shapes  

Although accuracy is identical, HOG provides a more structured, edge-based representation compared to raw intensity features.

---

# 🔬 Week 3 – Day 4: Robustness Analysis

To evaluate feature stability, experiments were conducted under:

- Gaussian Noise (Test Data Only)
- Brightness Shift (Test Data Only)

Training was performed on clean data. Only test data was modified.

---

## 🌫 Gaussian Noise Experiment

| Method        | Accuracy |
|--------------|----------|
| Raw Pixels   | 100%     |
| HOG Features | 76.6%    |

HOG Confusion Matrix:
[[63 0]
[29 32]]


### 🔎 Observation

- HOG performance degraded significantly under Gaussian noise.
- Gradient computation amplifies high-frequency pixel perturbations.
- Raw pixel representation remained stable due to strong class separability and small noise magnitude.

---

## 🌞 Brightness Shift Experiment

| Method        | Accuracy |
|--------------|----------|
| Raw Pixels   | 100%     |
| HOG Features | 100%     |

### 🔎 Observation

- Uniform brightness shifts did not affect classification.
- HOG is naturally illumination invariant (gradient cancels constant shift).
- Raw pixels also remained stable due to preserved class separability.

---

# 🎓 Key Insights

- HOG is robust to illumination changes and contrast scaling.
- HOG is sensitive to high-frequency pixel noise due to gradient amplification.
- Raw pixel features perform well on simple, clean datasets.
- Robustness depends on the type of perturbation applied.
- Feature engineering has limitations when moving toward real-world variability.

---

# 🚀 Future Improvements

- Evaluate robustness under rotation and scaling.
- Compare Logistic Regression with SVM.
- Apply data augmentation techniques.
- Transition toward CNN-based feature learning.
- Test on more complex and real-world datasets.

---

## 📌 Project Status

✔ Classical CV pipeline implemented  
✔ Feature engineering comparison completed  
✔ Robustness experiments conducted  
✔ Git-based version control integrated  

---


