import cv2
import numpy as np
import os

TARGET_SIZE = 128

def process_folder(folder_path, label):
    X = []
    Y = []

    print("Processing folder:", folder_path)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        print("Trying to open:", img_path)

        img = cv2.imread(img_path)

        if img is None:
            print("Failed to load image", img_path)
            continue

        h, w = img.shape[:2]

        # Scaling
        scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resizing
        img_resized = cv2.resize(img, (new_w, new_h))

        # Padding
        pad_w = TARGET_SIZE - new_w
        pad_h = TARGET_SIZE - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_top, pad_bottom,
            pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Gray Scale
        gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)

        # Normalizing
        normalized = gray.astype(np.float32) / 255.0

    return normalized

def process_single_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found or cannot be read")

    h, w = img.shape[:2]

    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    pad_w = TARGET_SIZE - new_w
    pad_h = TARGET_SIZE - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)

    normalized = gray.astype(np.float32) / 255.0

    return normalized
