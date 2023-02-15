import onnxruntime as ort
import time
import numpy as np
import cv2
import flow_utils


available_providers = ort.get_available_providers()
providers = []
if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")


def main(model_path: str,
         image1: str, image2: str, output_path: str):

    print(f"\033[32mReading model from file {model_path}\033[0m")
    sess = ort.InferenceSession(
        model_path,
        providers=providers,
    )
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [i.name for i in sess.get_outputs()]

    input_height = sess.get_inputs()[0].shape[2]
    input_width = sess.get_inputs()[0].shape[3]
    # print(f"{input_height=}")
    # print(f"{input_width=}")
    print(f"Input Shape: {sess.get_inputs()[0].shape}")
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    org_h, org_w = img1.shape[:2]
    img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), (input_width, input_height)).astype(np.float32) / 255.0
    img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), (input_width, input_height)).astype(np.float32) / 255.0

    img1 = np.transpose(img1, (2, 0, 1))[np.newaxis, :, :, :]
    img2 = np.transpose(img2, (2, 0, 1))[np.newaxis, :, :, :]

    for _ in range(5):
        t = time.time()
        outputs = sess.run(output_names,
            {
                input_names[0]: img1,
                input_names[1]: img2,
            }
        )
        dt = time.time() - t
        print(f"\033[34mElapsed: {dt:.3f} sec, {1/dt:.3f} FPS\033[0m")

    flow = outputs[0][0].transpose(1, 2, 0)
    drawn = flow_utils.flow_to_image(flow)
    drawn = cv2.resize(drawn, (org_w, org_h))
    cv2.imwrite(output_path, drawn)
    print(f"\033[32moutput: {output_path}\033[0m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="gmflow-scale1-mixdata-train320x576-4c3a6e9a_1x3x480x640_sim.onnx",
        help="ONNX model file path.")
    parser.add_argument(
        "-i1",
        "--image1",
        type=str,
        default="data/flow/frame1.png",
        help="input image1.")
    parser.add_argument(
        "-i2",
        "--image2",
        type=str,
        default="data/flow/frame2.png",
        help="input image2.")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output.png",
        help="output colored disparity image paht.")
    args = parser.parse_args()

    main(
        args.model_path,
        args.image1, args.image2, args.output_path
    )
