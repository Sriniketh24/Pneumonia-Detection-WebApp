import os
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

# --- Configuration ---
# Create a Flask web application
app = Flask(__name__)

# Load your trained Keras model
MODEL_PATH = 'pneumonia_detection_finetuned_fixed.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # We can still run the app to show the UI, but prediction will fail.
    model = None

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image dimensions for the model
IMG_SIZE = (224, 224)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    """Loads and preprocesses an image for model prediction."""
    try:
        # Load the image using Keras preprocessing
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        # Convert image to a NumPy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Rescale pixel values (must match training preprocessing)
        img_array = img_array / 255.0
        # Expand dimensions to create a batch of 1 image
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {img_path}: {e}")
        return None


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part in the request.')

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        # If the file is valid, process it
        if file and allowed_file(file.filename):
            # Secure the filename to prevent security issues
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image and get a prediction
            if model is None:
                return render_template('result.html', error="Model not loaded. Cannot make prediction.")

            processed_img = preprocess_image(filepath)
            if processed_img is not None:
                prediction = model.predict(processed_img)
                # The output of the sigmoid is a probability for the 'PNEUMONIA' class
                probability_pneumonia = prediction[0][0]

                # Interpret the prediction
                threshold = 0.5
                if probability_pneumonia >= threshold:
                    predicted_class = CLASS_NAMES[1]  # PNEUMONIA
                else:
                    predicted_class = CLASS_NAMES[0]  # NORMAL

                confidence = probability_pneumonia * 100 if predicted_class == 'PNEUMONIA' else (
                                                                                                            1 - probability_pneumonia) * 100

                # Render the result page
                return render_template('result.html',
                                       predicted_class=predicted_class,
                                       confidence=confidence,
                                       image_filename=filename)
            else:
                return render_template('index.html', error='Failed to process the image.')

    # For a GET request, just show the upload page
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded file to be displayed on the result page."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Main Application Execution ---
if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Run the Flask app
    # Use debug=True for development (it will auto-reload on code changes)
    app.run(debug=True)
