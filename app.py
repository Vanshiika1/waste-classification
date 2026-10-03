from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained model
model = load_model("model/waste_classifier_v4.h5")

# Waste classes
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

@app.route("/")
def home():
    return "Waste Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
