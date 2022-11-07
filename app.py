from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow.keras import models, layers


# Define a flask app
app = Flask(__name__)


#load model
from tensorflow.keras.models import load_model
new_model = load_model('models/model_2.h5')


@app.route('/')
def index():
    # Main page
    return render_template('index.html')#run the html file in th template folder


@app.route('/prediction', methods=['POST'])#if we run http://127.0.0.1:5000/predict this code under this will be executed
def prediction():
    
    img =request.files['img']
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    #request - a function in flask ['img] - name of the <input type="file"> in the index.html
    img.save("img.jpj")#img.jpj - we can give any name and the img will be saved in the location of this file(app.py)
    img1 = Image.open('img.jpj')#don't use opencv to read the image
    img1 = img1.resize((256,256))
    img_array = tf.keras.preprocessing.image.img_to_array(img1)
    img_array = tf.expand_dims(img_array,0)
    predictions = new_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    
    
    
    return render_template("index.html", data  = predicted_class)


if __name__ == '__main__':
    app.run(debug=True)

