import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import onnx.shape_inference
import onnxruntime.tools.symbolic_shape_infer
import onnxsim
import torch

from models.experimental import attempt_load
from utils.general import check_img_size


def resize_preserving_aspect_ratio(img: np.ndarray, img_size: int, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
    # Resize preserving aspect ratio. scale_ratio is the scaling ratio of the img_size.
    h, w = img.shape[:2]
    scale = img_size // scale_ratio / max(h, w)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img, scale


def transform_image(img: np.ndarray, img_size: int) -> torch.Tensor:
    img, _ = resize_preserving_aspect_ratio(img, img_size)

    pad = (0, img_size - img.shape[0], 0, img_size - img.shape[1])
    img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = torch.as_tensor(img, dtype=torch.float32)  # ndarray to Tensor, uint8 to fp16/32
    img = img[:, :, [2, 1, 0]].permute((2, 0, 1)).contiguous()  # BGR to RGB, HWC to CHW
    img /= 255  # 0~255 to 0~1
    img.unsqueeze_(0)  # add batch dimension
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s-face.pt', help='weights path')
    parser.add_argument('--img_size', type=int, default=640, help='image size')
    parser.add_argument('--dynamic', action='store_true', help='enable dynamic axis in onnx model')
    parser.add_argument('--skip_simplify', action='store_true', help='skip onnx-simplifier')
    # =======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', help='fp16 infer')
    # ================================================================
    args = parser.parse_args()
    print(args)

    # Create torch model
    model = attempt_load(args.weights, map_location=torch.device('cpu'))  # load FP32 model
    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid = [torch.zeros(1)] * 3  # nl=3 number of detection layers
    model.model[-1].export_cat = True
    model.eval()

    # Create input data
    img = cv2.imread('torch2trt/sample.jpg')
    img = transform_image(img, check_img_size(args.img_size, model.stride.max()))

    # Define output file path
    output_dir = 'onnx_files'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.weights).replace('.pt', '.onnx'))

    # Define input and output names
    input_names = ['img']
    output_names = ['pred']

    # Define dynamic_axes
    if args.dynamic:
        dynamic_axes = {input_names[0]: {0: 'N', 2: 'H', 3: 'W'},
                        output_names[0]: {0: 'N', 1: 'Candidates', 2: 'dyn_16'}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    torch.onnx.export(
        model,
        img,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        dynamic_axes=dynamic_axes,
    )

    # Check exported onnx model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    try:
        onnx_model = onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx_model)
        onnx.save(onnx_model, output_path)
    except Exception as e:
        print(f'ERROR: {e}, skip symbolic shape inference.')
    onnx.shape_inference.infer_shapes_path(output_path, output_path, check_type=True, strict_mode=True, data_prop=True)

    # Compare output with torch model and ONNX model
    torch_out = model(img).detach().numpy()
    session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        print(e)
        stdin = input('Do you want to ignore the error and proceed with the export ([y]/n)? ')
        if stdin == 'n':
            os.remove(output_path)
            exit(1)

    # Simplify ONNX model
    if not args.skip_simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: img.shape}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')

    # TensorRT export
    if args.onnx2trt:
        from torch2trt.trt_model import ONNX_to_TRT

        print('\nStarting TensorRT export...')
        ONNX_to_TRT(output_path, output_path.replace('.onnx', '.trt'), fp16_mode=args.fp16_trt)
