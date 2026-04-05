from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model("brain_tumor_type_model.h5")

# Classes
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Image path
img_path = "C:\\Users\\MUMMANA TRINADH\\Downloads\\brain tumor\\Testing\\glioma\\Te-gl_5.jpg"

# Load & preprocess image
img = image.load_img(img_path, target_size=(64,64))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
result = model.predict(img_array)
predicted_class_index = np.argmax(result)
predicted_class = classes[predicted_class_index]

# Tumor detection logic
if predicted_class != "No Tumor":
    print("Tumor Detected")                       # First line
    print("Tumor Type:", predicted_class)        # Second line
else:
    print("No Tumor Detected")                   # If no tumora