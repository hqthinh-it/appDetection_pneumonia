from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model('best_model.h5')

# Function to process input image
def process_image(img_path):
    # Load and resize the image to 100x100 pixels
    img = Image.open(img_path)
    img = img.resize((224, 224))
    
    # Convert the image to RGB if it's grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array


# Map class indices to human-readable labels
class_labels = {
    0: 'normal',
    1: 'pneumonia',
}

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    # Save the image temporarily
    img_path = 'static/temp/temp_img.jpg'
    file.save(img_path)
    
    # Process the image
    img_array = process_image(img_path)
    
    # Make a single prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    
    # Determine the label based on confidence
    if predictions[0][predicted_class] < 0.5:
        result = {'class': 'normal', 'confidence': float(predictions[0][predicted_class])}
    else:
        result = {'class': 'pneumonia', 'confidence': float(predictions[0][predicted_class])}

    # Render the template with the prediction result
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
