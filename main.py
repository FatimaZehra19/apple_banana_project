from xml.parsers.expat import model
from Preprocessing import process_folder
from Train import train_model
import numpy as np
from Preprocessing import process_folder, process_single_image
 
apple_X, apple_y = process_folder("data_set/Apple", label=0)
banana_X, banana_y = process_folder("data_set/Banana", label=1)

X = np.array(apple_X + banana_X)
y = np.array(apple_y + banana_y)

model = train_model(X, y)

# Test on a new image
test_image_path = "test_image.jpg"  

new_image = process_single_image(test_image_path)

prediction = model.predict([new_image])

if prediction[0] == 0:
    print("Prediction: Apple 🍎")
else:
    print("Prediction: Banana 🍌")
