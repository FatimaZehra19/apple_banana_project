import numpy as np

def add_gaussian_noise(image, mean=0, std=0.15):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure pixel values are valid
    return noisy_image