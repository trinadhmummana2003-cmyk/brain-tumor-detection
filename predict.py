import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("brain_tumor_type_model.keras")

classes = ['glioma','meningioma','notumor','pituitary']

def predict_tumor(img_path):

    # Load image
    img = image.load_img(img_path, target_size=(64,64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    tumor_type = classes[class_index]

    # Default values
    tumor_percentage = 0
    severity = "None"

    if tumor_type != "notumor":

        img_cv = cv2.imread(img_path, 0)
        img_cv = cv2.resize(img_cv, (256,256))

        _, thresh = cv2.threshold(img_cv,150,255,cv2.THRESH_BINARY)

        tumor_pixels = np.sum(thresh==255)
        total_pixels = thresh.size

        tumor_percentage = (tumor_pixels/total_pixels)*100

        if tumor_percentage <= 30:
            severity = "Simple"
        elif tumor_percentage <= 70:
            severity = "Moderate"
        else:
            severity = "High"

    # Suggestions
    if tumor_type == "glioma":
        doctor = "Consult neurologist immediately."
        food = "Berries, leafy greens, walnuts."
        lifestyle = "Good sleep, reduce stress."

    elif tumor_type == "meningioma":
        doctor = "Regular MRI monitoring."
        food = "Protein rich food."
        lifestyle = "Avoid smoking."

    elif tumor_type == "pituitary":
        doctor = "Consult endocrinologist."
        food = "Calcium rich food."
        lifestyle = "Maintain routine."

    else:
        doctor = "No tumor detected."
        food = "Balanced diet."
        lifestyle = "Exercise regularly."

    return tumor_type, round(tumor_percentage,2), severity, doctor, food, lifestyle
"""img_path ="C:\\Users\\MUMMANA TRINADH\\Downloads\\brain tumor\\Testing\\pituitary\\Te-pi_13.jpg"

tumor_type, percentage, severity, doctor, food, lifestyle = predict_tumor(img_path)

print("Tumor Type:", tumor_type)
print("Tumor Percentage:", percentage)
print("Severity:", severity)
print("Doctor Advice:", doctor)
print("Food Suggestion:", food)
print("Lifestyle:", lifestyle)"""