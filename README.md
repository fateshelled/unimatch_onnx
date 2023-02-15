# unimatch_onnx

ONNX or TensorRT inference demo for [Unimatch (Unifying Flow, Stereo and Depth Estimation)](https://github.com/autonomousvision/unimatch).

<img src="https://github.com/fateshelled/unimatch_onnx/blob/main/data/im0.png" width="320" height="240" alt=""><img src="https://user-images.githubusercontent.com/53618876/219047327-cf8db934-8603-4dd1-8af3-cfde86f01b69.png" width="320" height="240" alt="">

<img src="https://raw.githubusercontent.com/fateshelled/unimatch_onnx/main/data/left.png" width="320" height="240" alt=""><img src="https://user-images.githubusercontent.com/53618876/219047373-5358bca6-3912-4d33-a660-6cfd7264c2cf.png" width="320" height="240" alt="">


## Requirements
### ONNX model
- OpenCV
- numpy
- onnxruntime

※ tested onnxruntime==1.13.1

### TensorRT model
- OpenCV
- numpy
- TensorRT
- pycuda

※ tested TensorRT==8.5.2.2

## Model Download
[Google Drive](https://drive.google.com/drive/folders/1NtOPskzvVHoMQRT_a52QSlOFArGv8mFA)

## Usage
### ONNX model

#### Stereo Model
```
usage: demo_stereo_onnx.py [-h] [-m MODEL_PATH] [-l LEFT_IMAGE] [-r RIGHT_IMAGE] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        ONNX model file path. (default: unimatch_stereo_scale1_1x3x480x640_sim.onnx)
  -l LEFT_IMAGE, --left_image LEFT_IMAGE
                        input left image. (default: data/left.png)
  -r RIGHT_IMAGE, --right_image RIGHT_IMAGE
                        input right image. (default: data/right.png)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output colored disparity image paht. (default: output.png)
```

#### Opticalflow Model
```bash
usage: demo_flow_onnx.py [-h] [-m MODEL_PATH] [-i1 IMAGE1] [-i2 IMAGE2] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        ONNX model file path. (default: gmflow-scale1-mixdata-train320x576-4c3a6e9a_1x3x480x640_sim.onnx)
  -i1 IMAGE1, --image1 IMAGE1
                        input image1. (default: data/flow/frame1.png)
  -i2 IMAGE2, --image2 IMAGE2
                        input image2. (default: data/flow/frame2.png)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output colored disparity image paht. (default: output.png)
```

### TensorRT
- Before using the TensorRT demo, you will need to convert the onnx model file to an engine file for your GPU.
```bash
bash convert_onnx2trt.bash <onnx-model-path> <output-engine-path>
```

```bash
usage: demo_stereo_trt.py [-h] [-e ENGINE_PATH] [-ih INPUT_HEIGHT] [-iw INPUT_WIDTH] [-l LEFT_IMAGE] [-r RIGHT_IMAGE]
                          [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -e ENGINE_PATH, --engine_path ENGINE_PATH
                        TensorRT engine file path. (default: unimatch_stereo_scale1_1x3x480x640_sim.trt)
  -ih INPUT_HEIGHT, --input_height INPUT_HEIGHT
                        Model input height. (default: 480)
  -iw INPUT_WIDTH, --input_width INPUT_WIDTH
                        Model input width. (default: 640)
  -l LEFT_IMAGE, --left_image LEFT_IMAGE
                        input left image. (default: data/left.png)
  -r RIGHT_IMAGE, --right_image RIGHT_IMAGE
                        input right image. (default: data/right.png)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output colored disparity image paht. (default: output.png)
```
