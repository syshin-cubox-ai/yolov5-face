"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import os

import numpy as np
import onnx.shape_inference
import onnxruntime.tools.symbolic_shape_infer
import onnxsim
import torch
import torch.nn as nn

import models.common
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import check_img_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s-face.pt', help='weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='enable dynamic axis in onnx model')
    parser.add_argument('--skip_simplify', action='store_true', help='skip onnx-simplifier')
    # =======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', help='fp16 infer')
    # ================================================================
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid = [torch.zeros(1)] * 3  # nl=3 number of detection layers
    model.model[-1].export_cat = True
    model.eval()

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
    if not opt.dynamic:
        print(f'input_shape: {tuple(img.shape)}')

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
        if isinstance(m, models.common.ShuffleV2Block):  # shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()

    # Define output file path
    output_dir = 'onnx_files'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(opt.weights).replace('.pt', '.onnx'))

    # Define input and output names
    input_names = ['img']
    output_names = ['pred']

    # Define dynamic_axes
    if opt.dynamic:
        dynamic_axes = {input_names[0]: {0: 'N', 2: 'H', 3: 'W'},
                        output_names[0]: {0: 'N', 1: 'Candidates'}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    torch.onnx.export(
        model,
        img,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
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
    try:
        torch_out = model(img).numpy()
        session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        print(e)
        stdin = input('Do you want to ignore the error and proceed with the export ([y]/n)? ')
        if stdin == 'n':
            os.remove(output_path)
            exit(1)

    # Simplify ONNX model
    if not opt.skip_simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: img.shape}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')

    # TensorRT export
    if opt.onnx2trt:
        from torch2trt.trt_model import ONNX_to_TRT

        print('\nStarting TensorRT export...')
        ONNX_to_TRT(output_path, output_path.replace('.onnx', '.trt'), fp16_mode=opt.fp16_trt)
