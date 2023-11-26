from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# Ensure the 'static/uploads' directory exists
os.makedirs('static/uploads', exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/terms.html')  # Add this route
def terms():
    return render_template('terms.html')


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded file
    file_path = 'static/uploads/input.jpg'
    file.save(file_path)

    # Implement the colorization code here
    colorized_path = colorize_image(file_path)

    return render_template('result.html', input_image='input.jpg', colorized_image='colorized.jpg')


def colorize_image(image_path):
    prototxt_path = "models/colorization_deploy_v2.prototxt"
    model_path = "models/colorization_release_v2.caffemodel"
    kernel_path = "models/pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype("float32")]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(image_path)
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

    # Save the colorized image
    colorized_path = 'static/uploads/colorized.jpg'
    cv2.imwrite(colorized_path, colorized)

    return colorized_path


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)


if __name__ == '__main__':
    app.run(debug=True)
