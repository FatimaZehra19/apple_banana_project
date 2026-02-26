# 🍎 Apple vs Banana Image Classification  
**Classical Computer Vision → Deep Learning | Robustness & Invariance Study**

A structured computer vision project comparing handcrafted feature engineering and neural networks, including controlled robustness experiments under Gaussian noise and spatial translation.

---

## 🎯 Objective

- Compare Classical ML and Deep Learning models  
- Evaluate robustness to:
  - Gaussian noise (intensity corruption)
  - Spatial translation (object shift)
- Analyze the impact of data augmentation on invariance

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
- White background, centered objects  
- Train/Test split used (no data leakage)  
- Images resized to 128×128 grayscale  

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

# 📈 Unified Results Summary

| Model | Clean | Gaussian Noise (0.5) | 30% Translation | With Augmentation |
|--------|--------|----------------------|-----------------|-------------------|
| Logistic (Raw) | 100% | — | — | — |
| SVM (Raw/HOG) | 100% | — | — | — |
| FCNN | 100% | 78.2% | 70.92% | — |
| CNN (No Aug) | 99.5% | ~50% | 63.17% | — |
| CNN + Translation Aug | — | — | 98.22% | ✔ |

---

# 🧠 Key Insights

- The dataset is linearly separable under clean conditions.
- FCNN performs strongly on simple, globally separable data.
- CNN is not inherently translation-invariant — invariance must be learned.
- High-frequency Gaussian noise significantly impacts CNN without augmentation.
- Data augmentation dramatically improves robustness to the specific transformation seen during training.
- Robustness is transformation-specific (noise ≠ translation).

---

# 🎓 Major Takeaways

- Architecture alone does not guarantee robustness.
- Data augmentation is essential for real-world generalization.
- CNN advantages emerge under spatial variability.
- Experimental design and proper train/test separation are critical.

---

## 🚀 Project Status

✔ Classical CV pipeline  
✔ Feature engineering (Raw + HOG)  
✔ Logistic vs SVM comparison  
✔ FCNN implementation  
✔ CNN implementation  
✔ Gaussian noise analysis  
✔ Translation robustness study  
✔ Data augmentation experiments  
✔ Structured and reproducible workflow  

---

## 📌 Conclusion

This project demonstrates an end-to-end computer vision workflow — from handcrafted features to deep learning — with structured robustness evaluation and controlled experimentation.
s