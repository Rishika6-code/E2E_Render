import base64
import numpy as np
from io import BytesIO
# import os
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model


# -----------------------------
# Model configuration
# -----------------------------

IMG_SHAPE = (224, 224)
CLASSES = ["Modern", "Old"]

print("loading model.....")
model = load_model("fine_tuned_house.keras")
print("loaded model")


# -----------------------------
# Flask application
# -----------------------------
 
app = Flask(__name__)


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def sayhello():
    return render_template("index.html")


# -----------------------------
# Prediction route
# -----------------------------

@app.route("/predict-interior", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "img" not in request.files:
        return "No image uploaded", 400

    uploaded_file = request.files["img"]

    if uploaded_file.filename == "":
        return "No image selected", 400

    try:
        # --------------------------------
        # Read uploaded image ONCE
        # --------------------------------

        image_data = uploaded_file.read()

        # --------------------------------
        # Convert image to Base64
        # This is used to display the
        # uploaded image on the webpage
        # --------------------------------

        encoded_image = base64.b64encode(image_data).decode("utf-8")

        fpath = "data:image/png;base64," + encoded_image

        # --------------------------------
        # Open image from the same bytes
        # --------------------------------

        file = Image.open(BytesIO(image_data))

        # Convert RGBA/RGB/etc. to RGB
        file = file.convert("RGB")

        # --------------------------------
        # Resize image to model input size
        # --------------------------------

        file = file.resize(IMG_SHAPE)

        # --------------------------------
        # Convert image to NumPy array
        # --------------------------------

        file_array = np.asarray(file)

        # Shape should be:
        # (224, 224, 3)

        # --------------------------------
        # Prepare image for model
        # --------------------------------

        model_input = np.expand_dims(file_array, axis=0)

        # --------------------------------
        # Prediction
        # --------------------------------

        preds = model.predict(model_input)[0]

        i = np.argmax(preds)

        label = CLASSES[i]

        prob = preds[i]

        # --------------------------------
        # Prediction output
        # --------------------------------

        pred_output = {
            "img_size": file_array.shape,
            "label": label,
            "probability": np.round(prob * 100, 2)
        }

        # --------------------------------
        # Send result back to HTML
        # --------------------------------

        return render_template(
            "index.html",
            img_shape=file_array.shape,
            user_image=fpath,
            pred_output=pred_output
        )

    except Exception as e:

        print("Prediction error:", e)

        return f"Error while processing image: {str(e)}", 500


# -----------------------------
# Run Flask locally
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)