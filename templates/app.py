from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('breast_cancer_ultrasound_model.h5')

basepath = os.path.dirname(__file__)
upload_path = os.path.join(basepath, 'uploads')
if not os.path.exists(upload_path):
    os.makedirs(upload_path)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    name = request.form['name']
    age = request.form['age']
    symptoms = request.form['symptoms']
    history = request.form['history']
    
    if file:
        file_path = os.path.join(upload_path, file.filename)
        file.save(file_path)
        preds = model_predict(file_path, model)
        result = np.argmax(preds)
        
        diagnosis = "Positive for Breast Cancer" if result == 1 else "Negative for Breast Cancer"
        
        return render_template('result.html', name=name, age=age, symptoms=symptoms, history=history, prediction=diagnosis)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
