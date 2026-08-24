from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

# Define model and variables
IMG_SHAPE = (224, 224)
CLASSES = ["Modern", "Old"]

model = load_model("fine_tuned_house.keras")

# Define Flask app
app = Flask(__name__)

# Dummy result
res = {"Hello": "World"}

@app.route('/', methods=["GET"])
def intro():
    return jsonify(result=res)

@app.route('/predict-interior', methods=['POST'])
def predict():

    f = request.files['img']

     # Open image and convert RGBA/RGB/etc. to RGB
    file = Image.open(f).convert("RGB")
    file_shape = np.asarray(file).shape

    # Resize image to (224, 224) if needed
    if file.size != IMG_SHAPE:
        file = file.resize(IMG_SHAPE)
        file_shape = np.asarray(file).shape

    # Predictions
    preds = model.predict(np.expand_dims(file, axis=0))[0]
    i = np.argmax(preds)
    label = CLASSES[i]
    prob = preds[i]
    predictions = {
        "label": label,
        "prob": str(prob)
    }
    return jsonify(predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)