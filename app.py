from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('best_model.h5')  # Replace 'your_model.h5' with the actual name of your model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']
        img_path = f"uploads/{img.filename}"  # Assuming there's an 'uploads' folder for temporary storage
        img.save(img_path)

        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        prediction = model.predict(img_array)

        result = {'class': int(prediction[0][0] > 0.5), 'confidence': float(prediction[0][0])}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
