def extract_raw(image):
    # Placeholder for raw pixel extraction logic
    return image.flatten()


def extract_hog(image):
    from skimage.feature import hog

    features = hog(image, 
                   orientations=9, 
                   pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), 
                   block_norm='L2'
                   )
    return features