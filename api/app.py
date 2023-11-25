from flask import Flask, render_template, request, send_from_directory, url_for
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/terms.html')  
def terms():
    return render_template('terms.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Read the uploaded file into memory
    file_content = file.read()
    file_data = BytesIO(file_content)

    # Implement the colorization code here
    colorized_image = colorize_image(file_data)

    # Return the colorized image as a response
    return render_template('result.html', input_image='input.jpg', colorized_image=colorized_image)

def colorize_image(file_data):
    prototxt_path = get_absolute_path("api/models/colorization_deploy_v2.prototxt")
    model_path = get_absolute_path("api/models/colorization_release_v2.caffemodel")
    kernel_path = get_absolute_path("api/models/pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype("float32")]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Decode the image directly from memory
    image = cv2.imdecode(np.frombuffer(file_data.read(), np.uint8), cv2.IMREAD_COLOR)

    normalized = image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    # Encode the colorized image to JPEG format
    _, colorized_data = cv2.imencode('.jpg', colorized)
    
    # Return the colorized image as bytes
    return BytesIO(colorized_data).read()

if __name__ == '__main__':
    app.run(debug=True)
