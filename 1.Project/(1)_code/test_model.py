import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

np.set_printoptions(suppress=True)

# Load model
model = tf.keras.models.load_model(filepath='C:/Users/bitcamp/Downloads/converted_keras/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('C:/Users/bitcamp/Downloads/breed_final/Yorkshire_terrier/Z.jpg')

size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)

image.show()

normalized_img_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_img_array

predict = model.predict(data)
print(predict)