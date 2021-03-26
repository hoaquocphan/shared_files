import cv2
import numpy as np
import onnxruntime
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='dog.jpeg')
parser.add_argument('--m', type=str, default='model.onnx')
args = parser.parse_args()

def preprocess(img,w,h):   
    dim = (w,h)
    image = cv2.imread(args.image)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = input_img.reshape(1, 3, w, h)
    
    norm_img_data = np.zeros(input_img.shape).astype('float32')
    mean_vec = np.array([123.68, 116.779, 103.939])
    for i in range(input_img.shape[1]):
        norm_img_data[:,i,:,:] = input_img[:,i,:,:] - mean_vec[i]
    return input_img


def softmax(x): 
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def main(): 
    model_path = args.m
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    input_img = preprocess(args.image,input_shape[2],input_shape[3])
    raw_result = session.run([], {input_name: input_img})
    output_data = np.array(raw_result[0]).squeeze(axis=0)
    
    print(output_data)
    f = open("output_onnx.txt", "w")
    for i in output_data:
        f.write("\n")
        f.write(str(i))
    f.close()

    out_score = softmax(output_data)
    print("\n")
    print('score: {}'.format(np.amax(out_score)))
    
if __name__ == "__main__":
    main()