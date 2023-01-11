from flask import Flask, render_template, request, url_for, redirect, flash
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import base64
import io
from keras.models import load_model

config = {
    "DEBUG": False
}

app = Flask(__name__)
app.config.from_mapping(config)
app.config["SECRET_KEY"] = "coba123"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


Model_URL = os.getcwd()+'/static/models/'
Upload_URL = os.getcwd()+'/static/uploaded_img/'

np.set_printoptions(suppress=True)
model = load_model(Model_URL + 'keras_model.h5', compile=False)
class_names = open(Model_URL + 'labels.txt', 'r').readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def convert_img(input):
    img = Image.open(input)
    half = 0.5
    img = img.resize([int(half * s) for s in img.size])
    byte = io.BytesIO()
    img.save(byte, "PNG")
    img = base64.b64encode(byte.getvalue())
    return img.decode('utf-8')


def prediction(path):
    image = Image.open(path).convert('RGB')
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index] * 100
    confidence_score = round(confidence_score, 2)
    # print('Class:', class_name, end='')
    # return print('Confidence score:', confidence_score)
    hasil = dict()
    hasil['class'] = class_name
    hasil['skor'] = str(confidence_score)
    return hasil


@app.route("/", methods=['GET', 'POST'])
def home():
    data = {"title": "Klasifikasi Aksara Sunda Ngalagena Perhuruf", "page": "home"}
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('* No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('* No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(Upload_URL, filename))
            data['prediction'] = prediction(Upload_URL + filename)
            data['img'] = convert_img(Upload_URL + filename)
            os.unlink(os.path.join(Upload_URL, filename))
        else:
            flash('* Allowed image types are - jpg, jpeg, and png')
            return redirect(request.url)
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run()
