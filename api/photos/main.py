import numpy as np
import cv2


prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
image_path = 'photos/lion.jpg'

# Read neural network from caffe files
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# Load colorization points array
points = np.load(kernel_path)

# Modify points array
points = points.transpose().reshape(2, 313, 1, 1)
# Set as blobs for network layer
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype("float32")]
# Set layer parameters
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Read the image input
image = cv2.imread(image_path)
# Normalize pixel values
normalized = image.astype("float32") / 255.0
# Convert to LAB color space
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

# Resize
resized = cv2.resize(lab, (224, 224))
# Extract L channel
L = cv2.split(resized)[0]
# Adjust pixel values
L -= 50

# Set L channel
net.setInput(cv2.dnn.blobFromImage(L))
# Get predicted 'ab' channels
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
# Extract L channel
L = cv2.split(lab)[0]

# Combine L with ab
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
# Convert LAB to RGB
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
# Convert to uint8
colorized = (255.0 * colorized).astype("uint8")

# Display the original image
cv2.imshow("Original", image)
# Display the colorized image
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()