import onnxruntime as ort
import time
import numpy as np
import os
import cv2
from glob import glob

available_providers = ort.get_available_providers()
providers = []
if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")


def main(model_path: str,
         left_image: str, right_image: str, output_path: str):

    print(f"\033[32mReading model from file {model_path}\033[0m")
    sess = ort.InferenceSession(
        model_path,
        providers=providers,
    )
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [i.name for i in sess.get_outputs()]

    input_height = sess.get_inputs()[0].shape[2]
    input_width = sess.get_inputs()[0].shape[3]
    print(f"{input_height=}")
    print(f"{input_width=}")

    left = cv2.resize(cv2.cvtColor(cv2.imread(left_image), cv2.COLOR_BGR2RGB), (input_width, input_height)).astype(np.float32) / 255.0
    right = cv2.resize(cv2.cvtColor(cv2.imread(right_image), cv2.COLOR_BGR2RGB), (input_width, input_height)).astype(np.float32) / 255.0

    left = np.transpose(left, (2, 0, 1))[np.newaxis, :, :, :]
    right = np.transpose(right, (2, 0, 1))[np.newaxis, :, :, :]

    for _ in range(5):
        t = time.time()
        output = sess.run(output_names,
            {
                input_names[0]: left,
                input_names[1]: right,
            }
        )
        dt = time.time() - t
        print(f"\033[34mElapsed: {dt:.3f} sec, {1/dt:.3f} FPS\033[0m")

    disp = output[0][0]
    norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    cv2.imwrite(output_path, colored)
    print(f"\033[32moutput: {output_path}\033[0m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="unimatch_stereo_scale1_1x3x480x640_sim.onnx",
        help="ONNX model file path.")
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
        args.model_path,
        args.left_image, args.right_image, args.output_path
    )
