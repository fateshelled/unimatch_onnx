import onnxruntime as ort
import time
import numpy as np
import cv2
import flow_utils
import os


available_providers = ort.get_available_providers()
providers = []
if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")


def main(model_path: str,
         image1: str, image2: str, output_path: str,
         bidir_flow: bool):

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
    img1 = cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2RGB)
    org_h, org_w = img1.shape[:2]
    img1 = cv2.resize(img1, (input_width, input_height)).astype(np.float32)
    img2 = cv2.resize(img2, (input_width, input_height)).astype(np.float32)

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

    output = outputs[0]
    forward_flow = output[0].transpose(1, 2, 0)
    if input_height != org_h or input_width != org_w:
        forward_flow = cv2.resize(forward_flow, (org_w, org_h))
        forward_flow[:, :, 0] *= org_w / input_width
        forward_flow[:, :, 1] *= org_h / input_height
    drawn = flow_utils.flow_to_image(forward_flow)
    cv2.imwrite(output_path, drawn)
    print(f"\033[32moutput: {output_path}\033[0m")

    if bidir_flow:
        backward_flow = output[1].transpose(1, 2, 0)
        if input_height != org_h or input_width != org_w:
            backward_flow = cv2.resize(backward_flow, (org_w, org_h))
            backward_flow[:, :, 0] *= org_w / input_width
            backward_flow[:, :, 1] *= org_h / input_height
        drawn = flow_utils.flow_to_image(backward_flow)
        file, ext = os.path.splitext(output_path)
        cv2.imwrite(file + "_backward" + ext, drawn)
        print(f"\033[32moutput backward: {output_path}\033[0m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="gmflow-scale1-mixdata-train320x576-4c3a6e9a_1x3x480x640_sim.onnx",
        # default="gmflow-scale1-things-e9887eda_1x3x480x640_bidir_flow_sim.onnx",
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
    parser.add_argument(
        "-b",
        "--bidir_flow",
        # action="store_false",
        action="store_true",
        help="output colored disparity image paht.")
    args = parser.parse_args()

    main(
        args.model_path,
        args.image1, args.image2, args.output_path,
        args.bidir_flow
    )
