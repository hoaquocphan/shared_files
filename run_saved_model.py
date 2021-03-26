import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np #path of the directory where you want to save your model
import argparse
import os
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from PIL import Image
print(tf.version.VERSION)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='dir')
args = parser.parse_args()

PATH_TO_SAVED_MODEL = args.dir


def softmax(x): 
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

image_np = load_image_into_numpy_array("dog.jpeg")


# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)


input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32, saturate=False)

# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = model(input_tensor)

print("result:")

#num_detections = int(detections.pop('num_detections'))

detections = detections.numpy()

detections = detections.squeeze(axis=0)

print(detections)
f = open("output_savedmodel.txt", "w")
for i in detections:
    f.write("\n")
    f.write(str(i))
f.close()

out_score = softmax(detections)
print("\n")
print('score: {}'.format(np.amax(out_score)))



