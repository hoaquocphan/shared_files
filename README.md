
1/ run command to convert saved model to onnx model:
python3 -m tf2onnx.convert --saved-model imagenet_mobilenet_v1_025_128_classification_4/ --output imagenet_mobilenet_v1_025_128_classification_4.onnx --opset 12 --concrete_function 2 --inputs inputs:0[1,416,416,3] --inputs-as-nchw  inputs:0

2/ run command to test output of model
python3 run_onnx_model.py --m imagenet_mobilenet_v1_025_128_classification_4.onnx --image cat.png
python3 run_saved_model.py --dir imagenet_mobilenet_v1_025_128_classification_4 --image cat.png

