# 🍎 Apple vs Banana Image Classification  
**Classical Computer Vision → Deep Learning | Robustness & Invariance Study**

A structured computer vision project comparing handcrafted feature engineering and neural networks, with controlled robustness experiments under Gaussian noise and spatial translation.

---

## 🎯 Objective

- Compare classical ML and deep learning approaches.
- Evaluate robustness under:
  - Gaussian noise (intensity corruption)
  - Spatial translation (object shift)
- Analyze the effect of data augmentation on model invariance.

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
- Centered objects
- Controlled lighting

---

# 🔬 Models Implemented

### Classical Machine Learning
- Raw Pixel Features (16384-dim)
- HOG Features (8100-dim)
- Logistic Regression
- Linear SVM

### Deep Learning
- Fully Connected Neural Network (FCNN)
- Convolutional Neural Network (CNN)
- CNN with Data Augmentation

---

# 📈 Clean Data Performance

| Model | Accuracy |
|--------|----------|
| Logistic (Raw) | 100% |
| SVM (Raw) | 100% |
| FCNN | 100% |
| CNN | 99.5% |

Dataset is linearly separable under controlled conditions.

---

# 🌫 Gaussian Noise Robustness (Test Only)

## FCNN

| Noise Std | Accuracy |
|------------|----------|
| 0.1 | 100% |
| 0.2 | 100% |
| 0.3 | 100% |
| 0.4 | 97.7% |
| 0.5 | 78.2% |

## CNN (No Augmentation)

| Noise Std | Accuracy |
|------------|----------|
| 0.1 | 98.4% |
| 0.2 | 63.9% |
| 0.3 | 51.7% |
| ≥0.4 | ~50% |

Observation:
- CNN edge detectors degrade quickly under high-frequency noise.
- FCNN is more stable on this simple dataset due to strong global separability.

---

# 🔁 Translation Robustness (30% Horizontal Shift)

## FCNN
Accuracy: **70.92%**

## CNN (No Translation Training)
Accuracy: **63.17%**

## CNN + Translation Augmentation (Training with 30% Shift)
Accuracy: **98.22%**

---

# 🧠 Key Insights

- CNN is not inherently translation-invariant; invariance must be learned.
- Data augmentation dramatically improves robustness to the specific transformation used during training.
- Robustness is transformation-specific:
  - Rotation augmentation → rotation tolerance
  - Translation augmentation → shift tolerance
  - Noise augmentation required for noise robustness
- Architecture advantage depends on dataset complexity.

---

# 🎓 Major Takeaways

- Simple datasets can favor global models (FCNN).
- CNN advantages emerge under spatial variability.
- Data augmentation is critical for real-world generalization.
- Robustness depends on training exposure, not just architecture.

---

## 🚀 Project Status

✔ Classical CV pipeline  
✔ Logistic vs SVM comparison  
✔ FCNN implementation  
✔ CNN implementation  
✔ Gaussian noise analysis  
✔ Translation robustness study  
✔ Data augmentation experiments  
✔ Structured experimental workflow  

---