# unimatch_onnx

ONNX or TensorRT inference demo for [Unimatch (Unifying Flow, Stereo and Depth Estimation)](https://github.com/autonomousvision/unimatch).

※ Supported only stereo model now.

![stereo disparity](https://user-images.githubusercontent.com/53618876/218768500-db3aeda2-1475-4f31-8301-b599753ac749.png)

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
