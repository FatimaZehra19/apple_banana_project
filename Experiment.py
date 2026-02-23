from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score
from Preprocessing import process_folder, process_single_image
from Features import extract_raw, extract_hog
from Train import train_model
from utils import add_Gaussian_noise as add_noise
import numpy as np 
import time
import os

images = []
labels = []

# Example dataset loading loop
for label, folder in enumerate(["data_set/Apple", "data_set/Banana"]):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        
        # Preprocess
        img = process_single_image(path)
        
        images.append(img)
        labels.append(label)

print("===== BASELINE (CLEAN DATA) =====")

# Split clean images
X_train_img, X_test_img, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# -------- RAW Experiment --------

X_train_raw = [extract_raw(img) for img in X_train_img] 
X_test_raw  = [extract_raw(img) for img in X_test_img]

model_raw = LogisticRegression(max_iter=1000)
start_time = time.time()
model_raw.fit(X_train_raw, y_train)

y_pred_raw = model_raw.predict(X_test_raw) 
end_time = time.time()
time_raw = end_time - start_time
print("---- RAW PIXELS ----")
print("Feature length:", len(X_train_raw[0]))
print("Accuracy:", accuracy_score(y_test, y_pred_raw))
print("Training time:", time_raw)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_raw))


# -------- HOG Experiment --------
X_train_hog = [extract_hog(img) for img in X_train_img]
X_test_hog  = [extract_hog(img) for img in X_test_img]

model_hog = LogisticRegression(max_iter=1000)
start_time = time.time()
model_hog.fit(X_train_hog, y_train)

y_pred_hog = model_hog.predict(X_test_hog)
end_time = time.time()
time_hog = end_time - start_time


print("\n---- HOG FEATURES ----")
print("Feature length:", len(X_train_hog[0]))
print("Accuracy:", accuracy_score(y_test, y_pred_hog))
print("Training time:", time_hog)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_hog))


# Test on a new noisy image
print("\n\n===== NOISY TEST DATA EXPERIMENT =====")

# Add noise ONLY to test images
X_test_noisy = [add_noise(img) for img in X_test_img]

# ---- RAW (Train clean, test noisy) ----
X_train_raw = [extract_raw(img) for img in X_train_img]
X_test_raw_noisy = [extract_raw(img) for img in X_test_noisy]

model_raw.fit(X_train_raw, y_train)
y_pred_raw_noisy = model_raw.predict(X_test_raw_noisy)

print("---- RAW PIXELS (NOISY TEST) ----")
print("Accuracy:", accuracy_score(y_test, y_pred_raw_noisy))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_raw_noisy))

# ---- HOG (Train clean, test noisy) ----
X_train_hog = [extract_hog(img) for img in X_train_img]
X_test_hog_noisy = [extract_hog(img) for img in X_test_noisy]

model_hog.fit(X_train_hog, y_train)
y_pred_hog_noisy = model_hog.predict(X_test_hog_noisy)

print("\n---- HOG FEATURES (NOISY TEST) ----")
print("Accuracy:", accuracy_score(y_test, y_pred_hog_noisy))    
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_hog_noisy))



def add_brightness(image, factor=0.1):
    bright_image = image * factor
    bright_image = np.clip(bright_image, 0, 1)  # Ensure pixel values are valid
    return bright_image

print("\n\n===== BRIGHTNESS TEST DATA EXPERIMENT =====")

# Add brightness ONLY to test images
X_test_bright = [add_brightness(img) for img in X_test_img]

# ---- RAW (Train clean, test bright) ----
X_test_raw_bright = [extract_raw(img) for img in X_test_bright]
model_raw.fit(X_train_raw, y_train)
y_pred_raw_bright = model_raw.predict(X_test_raw_bright)

print("---- RAW PIXELS (BRIGHT TEST) ----")
print("Accuracy:", accuracy_score(y_test, y_pred_raw_bright))   
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_raw_bright))

# ---- HOG (Train clean, test bright) ----
X_test_hog_bright = [extract_hog(img) for img in X_test_bright]
model_hog.fit(X_train_hog, y_train)
y_pred_hog_bright = model_hog.predict(X_test_hog_bright)

print("\n---- HOG FEATURES (BRIGHT TEST) ----")         
print("Accuracy:", accuracy_score(y_test, y_pred_hog_bright))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_hog_bright))
