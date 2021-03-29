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
parser.add_argument('--image', type=str, default='dog.jpeg')
args = parser.parse_args()

PATH_TO_SAVED_MODEL = args.dir


def softmax(x): 
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def preprocess_2(img):
    image_np = np.array(Image.open(img))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32, saturate=False)
    input_tensor = np.expand_dims(input_tensor, 0)
    #input_tensor = np.moveaxis(input_tensor, 3, 1) #move axis at location 3 to location 1 
    #input_tensor = np.transpose(input_tensor, [0, 3, 1, 2])
    return input_tensor

model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

input_tensor = preprocess_2(args.image)

detections = model(input_tensor)

detections = detections.numpy()

detections = detections.squeeze(axis=0)

print("result:")
print(detections)
f = open("output_savedmodel.txt", "w")
for i in detections:
    f.write("\n")
    f.write(str(i))
f.close()

out_score = softmax(detections)
print("\n")
print('score: {}'.format(np.amax(out_score)))
print('item id: {}'.format(np.where(out_score == np.amax(out_score))[0][0]))



