# 🍎 Apple vs Banana Image Classification  
**Classical Computer Vision → Deep Learning | Robustness Evaluation**

A structured computer vision project comparing handcrafted feature engineering with neural network–based learning under controlled robustness experiments.

---

## 🎯 Objective

Develop and compare multiple classification approaches for apple vs banana image recognition, and evaluate robustness under Gaussian noise perturbations.

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- NumPy  
- Scikit-learn  
- PyTorch  

---

## 📊 Dataset

- ~620 images (Apple & Banana)  
- Clean white background  
- Controlled lighting  
- Centered objects  
- Dataset excluded from repository  

---

## 🔬 Models Implemented

### Classical Machine Learning
- Raw Pixel Features (16384-dim)
- HOG Features (8100-dim)
- Logistic Regression
- Linear SVM

### Deep Learning
- Fully Connected Neural Network (FCNN)  
  Architecture:  
  Input (128×128 grayscale) → Flatten → Linear(16384→128) → ReLU → Linear(128→2)  
  ~2M parameters

---

## 📈 Performance Summary

### Clean Dataset Accuracy

| Model | Feature | Accuracy |
|--------|----------|----------|
| Logistic | Raw | 100% |
| Logistic | HOG | 100% |
| SVM | Raw | 100% |
| SVM | HOG | 100% |
| FCNN | Raw Pixels | 100% |

The dataset is linearly separable under controlled conditions.

---

## 🌫 Gaussian Noise Robustness (Test Only)

FCNN evaluated under increasing Gaussian noise:

| Noise Std | Accuracy |
|------------|----------|
| 0.0 | 100% |
| 0.3 | 100% |
| 0.4 | 97.9% |
| 0.5 | 80.45% |

### Key Findings

- HOG degrades significantly under high-frequency noise.
- Raw pixel models remain stable under small perturbations.
- FCNN memorizes simple patterns but lacks spatial inductive bias.
- Performance drops as signal-to-noise ratio decreases.

---

## 🧠 Key Insights

- Feature representation impacts robustness more than classifier choice.
- Handcrafted gradients are sensitive to pixel-level noise.
- Fully connected networks do not preserve spatial structure.
- Spatially-aware architectures (CNN) are required for real-world generalization.

---

## 🚀 Next Phase

- Implement Convolutional Neural Network (CNN)
- Compare robustness vs FCNN
- Introduce data augmentation (translation, rotation)
- Evaluate generalization on more complex datasets

---

## 📌 Project Status

✔ Classical CV pipeline  
✔ Feature engineering comparison  
✔ Logistic vs SVM analysis  
✔ FCNN implementation  
✔ Robustness evaluation  
✔ Modular training & evaluation workflow  