import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model once
model = load_model("brain_tumor_type_model.keras")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


def get_age_group(age):
    if age is None:
        return "unknown"
    if 1 <= age <= 10:
        return "child"
    elif 11 <= age <= 18:
        return "teen"
    elif 19 <= age <= 40:
        return "adult"
    elif 41 <= age <= 60:
        return "middle_age"
    else:
        return "senior"


def get_personalized_guidance(tumor_type, age=None, gender=None):
    age_group = get_age_group(age)
    gender = str(gender).strip().lower() if gender else "unknown"

    # Base guidance by tumor type
    if tumor_type == "glioma":
        doctor = "Consult a neurologist or neurospecialist as early as possible."
        food = "Include leafy greens, berries, walnuts, and hydration-rich foods."
        lifestyle = "Focus on good sleep, stress reduction, and avoiding fatigue."

    elif tumor_type == "meningioma":
        doctor = "Regular neurological evaluation and follow-up scans are recommended."
        food = "Prefer balanced protein-rich meals with fresh vegetables and fruits."
        lifestyle = "Avoid smoking, maintain rest, and follow a regular routine."

    elif tumor_type == "pituitary":
        doctor = "Consult an endocrinologist and neurologist for detailed evaluation."
        food = "Include calcium-rich foods, protein, and nutrient-balanced meals."
        lifestyle = "Maintain sleep routine, light activity, and proper hydration."

    else:
        doctor = "No tumor pattern detected. Continue routine health monitoring if needed."
        food = "Maintain a balanced diet with fruits, vegetables, and enough water."
        lifestyle = "Exercise regularly, sleep well, and manage daily stress."

    # Age-based support note
    if age_group == "child":
        age_note = "Child care note: ensure parental supervision, balanced meals, hydration, and proper sleep."
    elif age_group == "teen":
        age_note = "Teen care note: reduce screen stress, maintain sleep schedule, and support nutrition."
    elif age_group == "adult":
        age_note = "Adult care note: maintain work-life balance, regular meals, and adequate rest."
    elif age_group == "middle_age":
        age_note = "Middle-age care note: monitor stress, routine health checkups, and controlled lifestyle habits."
    elif age_group == "senior":
        age_note = "Senior care note: prioritize regular checkups, light activity, hydration, and assisted care if needed."
    else:
        age_note = "Age-specific care note unavailable."

    # Gender-based support note
    if gender == "male":
        gender_note = "Male health note: maintain regular health monitoring, sleep quality, and stress control."
    elif gender == "female":
        gender_note = "Female health note: maintain nutrition, hydration, rest, and regular medical follow-up when needed."
    else:
        gender_note = "General health note: follow balanced nutrition, rest, and medical guidance."

    return doctor, food, lifestyle, age_note, gender_note


def predict_tumor(img_path, age=None, gender=None):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    tumor_type = classes[class_index]

    # Confidence
    confidence = float(np.max(prediction) * 100)

    # Tumor detected or not
    tumor_detected = tumor_type != "notumor"

    tumor_percentage = 0.0
    severity = "None"

    if tumor_detected:
        img_cv = cv2.imread(img_path, 0)
        img_cv = cv2.resize(img_cv, (256, 256))

        _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)

        tumor_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size

        tumor_percentage = (tumor_pixels / total_pixels) * 100

        if tumor_percentage <= 30:
            severity = "Low"
        elif tumor_percentage <= 70:
            severity = "Moderate"
        else:
            severity = "High"

    doctor, food, lifestyle, age_note, gender_note = get_personalized_guidance(
        tumor_type, age=age, gender=gender
    )

    return {
        "tumor_detected": tumor_detected,
        "tumor_status": "Detected" if tumor_detected else "Not Detected",
        "tumor_type": "No Tumor" if tumor_type == "notumor" else tumor_type.capitalize(),
        "confidence": round(confidence, 2),
        "tumor_percentage": round(tumor_percentage, 2),
        "severity": severity,
        "doctor": doctor,
        "food": food,
        "lifestyle": lifestyle,
        "age_note": age_note,
        "gender_note": gender_note,
    }