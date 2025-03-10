from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
import firebase_admin
from firebase_admin import credentials, auth
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.preprocessing import image
import os

# Initialize Firebase

load_dotenv()

# Flask App Initialization
app = Flask(__name__)

# Use SECRET_KEY from .env
app.secret_key = os.getenv("SECRET_KEY")

app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize Firebase using the credential path from .env
firebase_config_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate(firebase_config_path)
firebase_admin.initialize_app(cred)


# Load ML Model
model = tf.keras.models.load_model('my_model.h5')
class_labels = ['0', '1', '2', '3', '4']

def load_and_preprocess_image(img_path, target_size=(150, 150, 3)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_disease_details(class_name):
    disease_info = {
        '0': {"name": "No Disease", "description": "The eye appears healthy with no signs of disease."},
        '1': {
            "name": "Diabetic Retinopathy",
            "description": "Damage to the retina caused by high blood sugar levels.",
            "symptoms": {
                "early": "None",
                "later": "Blurry vision, floating spots in your vision, blindness"
            },
            "diagnosis": "Dilated eye exam",
            "treatment": "Injections, laser treatment, surgery"
        },
        '2': {
            "name": "Cataract",
            "description": "Clouding of the lens, leading to blurred or impaired vision.",
            "symptoms": {
                "early": "None",
                "later": "Blurry vision, colors that seem faded, sensitivity to light, trouble seeing at night, double vision"
            },
            "diagnosis": "Dilated eye exam",
            "treatment": "Surgery"
        },
        '3': {
            "name": "Glaucoma",
            "description": "A group of eye conditions damaging the optic nerve, leading to vision loss.",
            "symptoms": {
                "early": "Often none",
                "later": "Loss of side (peripheral) vision, blind spots, blindness"
            },
            "diagnosis": "Dilated eye exam with visual field testing",
            "treatment": "Medicine (usually eye drops), laser treatment, surgery"
        },
        '4': {
            "name": "Other Eye Conditions",
            "description": "Unclassified conditions requiring further analysis.",
            "symptoms": {"early": "Varies", "later": "Varies"},
            "diagnosis": "Consultation with an ophthalmologist",
            "treatment": "Varies depending on condition"
        }
    }

    # Ensure a default dictionary is returned if class_name is not found
    return disease_info.get(class_name, {"name": "Unknown", "description": "Details not available."})


import logging

logging.basicConfig(level=logging.DEBUG)

def predict_disease(model, img_path, class_labels):
    logging.debug(f"Loading and preprocessing image: {img_path}")
    img_array = load_and_preprocess_image(img_path)

    logging.debug("Running prediction")
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    logging.debug(f"Prediction result: {predicted_class} with confidence {confidence:.2f}%")

    # Fetch disease details
    disease_details = get_disease_details(predicted_class)
    if disease_details:
        disease_details["confidence"] = f"{confidence:.2f}%"
        disease_details["image"] = img_path
    else:
        raise ValueError(f"No details found for class: {predicted_class}")

    return disease_details


# Routes
@app.route('/')
def home():
    user = session.get("user")
    if user:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')  # Safely get 'email'
        password = request.form.get('password')  # Safely get 'password'

        # Validate inputs
        if not email or not password:
            return "Email and password are required!", 400

        # Perform login logic here (e.g., authenticate with Firebase or a database)
        session['user'] = email
        return redirect(url_for('upload'))

    return render_template('login.html')


@app.route('/login/google', methods=['POST'])
def login_with_google():
    id_token = request.form['id_token']  # Token from client-side Google Sign-In
    try:
        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token['email']
        session['user'] = email
        return jsonify({"success": True, "redirect": url_for('upload')})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user(email=email, password=password)
            session['user'] = email
            return redirect(url_for('upload'))
        except Exception as e:
            return render_template('signup.html', error="Error creating user: " + str(e))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    user = session.get("user")
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file uploaded!", user=user)

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No file selected!", user=user)

        # Ensure 'static/styles' directory exists
        upload_folder = 'static/Uploads'
        os.makedirs(upload_folder, exist_ok=True)

        img_path = os.path.join(upload_folder, file.filename)
        file.save(img_path)

        try:
            prediction = predict_disease(model, img_path, class_labels)
            return render_template('result.html', result=prediction, img_path=img_path, user=user)
        except Exception as e:
            return render_template('upload.html', error=f"Prediction failed: {str(e)}", user=user)

    return render_template('upload.html', user=user)

if __name__ == '__main__':
    app.run(debug=True)
