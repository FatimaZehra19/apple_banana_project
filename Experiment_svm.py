import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from Preprocessing import process_single_image
from Features import extract_raw, extract_hog
from utils import add_gaussian_noise 


# Load dataset
images = []
labels = []

for label in os.listdir('data_set'):
    for img_file in os.listdir(os.path.join('data_set', label)):
        img_path = os.path.join('data_set', label, img_file)
        images.append(process_single_image(img_path))
        labels.append(label)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


print("===========SVM Experiment===============")

# Raw + SVM
X_train_raw = [extract_raw(img) for img in X_train]
X_test_raw = [extract_raw(img) for img in X_test]

start_time = time.time()
model_raw_svm = SVC(kernel='linear')
model_raw_svm.fit(X_train_raw, y_train)

training_time_raw_svm = time.time() - start_time
y_pred_raw_svm = model_raw_svm.predict(X_test_raw)

print("\n-------Raw + SVM---------")
print("feature length:", len(X_train_raw[0]))
print("Training time:", training_time_raw_svm)
print("Accuracy:", accuracy_score(y_test, y_pred_raw_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_raw_svm))  

# HOG + SVM         
X_train_hog = [extract_hog(img) for img in X_train]
X_test_hog = [extract_hog(img) for img in X_test]
start_time = time.time()

model_hog_svm = SVC(kernel='linear')
model_hog_svm.fit(X_train_hog, y_train)

training_time_hog_svm = time.time() - start_time
y_pred_hog_svm = model_hog_svm.predict(X_test_hog)

print("\n-------HOG + SVM---------")
print("feature leangth:", len(X_train_hog[0]))      
print("Training time:", training_time_hog_svm)
print("Accuracy:", accuracy_score(y_test, y_pred_hog_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_hog_svm))

# Experiment with noise
print("\n===========SVM with Noise===============")

X_test_raw_noisy = [extract_raw(add_gaussian_noise(img)) for img in X_test_img]
y_pred_raw_svm_noisy = model_raw_svm.predict(X_test_raw_noisy)

print("\n-------Raw + SVM with Noise---------")
print("Accuracy:", accuracy_score(y_test, y_pred_raw_svm_noisy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_raw_svm_noisy))

X_test_hog_noisy = [extract_hog(add_gaussian_noise(img)) for img in X_test_img]
y_pred_hog_svm_noisy = model_hog_svm.predict(X_test_hog_noisy)

print("\n-------HOG + SVM with Noise---------")
print("Accuracy:", accuracy_score(y_test, y_pred_hog_svm_noisy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_hog_svm_noisy))

