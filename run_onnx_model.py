import cv2
import numpy as np
import onnxruntime
import argparse
from PIL import Image
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='dog.jpeg')
parser.add_argument('--m', type=str, default='model.onnx')
args = parser.parse_args()

def preprocess_1(img,w,h):   
    dim = (w,h)
    image = cv2.imread(img)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = input_img.reshape(1, 3, w, h)
    input_img = input_img/255
    return input_img

def preprocess_2(img):
    image_np = np.array(Image.open(img))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32, saturate=False)
    input_tensor = np.expand_dims(input_tensor, 0)
    #inp = np.moveaxis(input_tensor, 3, 1) #move axis at location 3 to location 1 
    inp = np.transpose(input_tensor, [0, 3, 1, 2])
    return inp

def softmax(x): 
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def main(): 
    model_path = args.m
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    #inp = preprocess_1(args.image,input_shape[2],input_shape[3])
    inp = preprocess_2(args.image)

    raw_result = session.run(None, {input_name: inp})
    output_data = np.array(raw_result[0]).squeeze(axis=0)
    
    print("result:")
    print(output_data)
    f = open("output_onnx.txt", "w")
    for i in output_data:
        f.write("\n")
        f.write(str(i))
    f.close()

    out_score = softmax(output_data)
    print("\n")
    print('score: {}'.format(np.amax(out_score)))
    print('item id: {}'.format(np.where(out_score == np.amax(out_score))[0][0]))
    
if __name__ == "__main__":
    main()