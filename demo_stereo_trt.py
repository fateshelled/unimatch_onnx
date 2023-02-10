import time
import numpy as np
import tensorrt as trt
import os
import cv2
from glob import glob
import trt_common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def get_engine(engine_file_path):
    print(f"\033[32mReading engine from file {engine_file_path}\033[0m")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main(engine_path: str, input_height: int, input_width: int,
         left_image: str, right_image: str, output_path: str):

    engine = get_engine(engine_path)
    context = engine.create_execution_context()

    left = cv2.resize(cv2.cvtColor(cv2.imread(left_image), cv2.COLOR_BGR2RGB), (input_width, input_height)).astype(np.float32) / 255.0
    right = cv2.resize(cv2.cvtColor(cv2.imread(right_image), cv2.COLOR_BGR2RGB), (input_width, input_height)).astype(np.float32) / 255.0

    left = np.transpose(left, (2, 0, 1))[np.newaxis, :, :, :]
    right = np.transpose(right, (2, 0, 1))[np.newaxis, :, :, :]

    for _ in range(5):
        t = time.time()
        inputs, outputs, bindings, stream = trt_common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(left)
        inputs[1].host = np.ascontiguousarray(right)
        outputs = trt_common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        dt = time.time() - t
        print(f"\033[34mElapsed: {dt:.3f} sec, {1/dt:.3f} FPS\033[0m")

    disp = outputs[0].reshape(input_height, input_width)
    norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    cv2.imwrite(output_path, colored)
    print(f"\033[32moutput: {output_path}\033[0m")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e",
        "--engine_path",
        type=str,
        default="unimatch_stereo_scale1_1x3x480x640_sim.trt",
        help="TensorRT engine file path.")
    parser.add_argument(
        "-ih",
        "--input_height",
        type=int,
        default=480,
        help="Model input height.")
    parser.add_argument(
        "-iw",
        "--input_width",
        type=int,
        default=640,
        help="Model input width.")
    parser.add_argument(
        "-l",
        "--left_image",
        type=str,
        default="data/left.png",
        help="input left image.")
    parser.add_argument(
        "-r",
        "--right_image",
        type=str,
        default="data/right.png",
        help="input right image.")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output.png",
        help="output colored disparity image paht.")
    args = parser.parse_args()

    main(
        args.engine_path, args.input_height, args.input_width,
        args.left_image, args.right_image, args.output_path
    )
