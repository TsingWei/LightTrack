import _init_paths
from lib.models.models import LightTrackM_Speed
import torch
import time
import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16
import onnx
from onnxsim import simplify as simplify_func

if __name__ == "__main__":
    # test the running speed
    path_name = 'back_04502514044521042540+cls_211000022+reg_100000111_ops_32'  # our 530M model
    use_gpu = False
    # torch.cuda.set_device(0)
    model = LightTrackM_Speed(path_name=path_name)
    x = torch.randn(1, 3, 256, 256)
    zf = torch.randn(1, 96, 8, 8)

    dtype = np.float16
    inputs=( x,zf)
    inputs_onnx = {'zf':  np.array(zf.cpu(), dtype=dtype),
                   'x': np.array(x.cpu(), dtype=dtype),
                   }
    # output_onnx_name = 'test_net.onnx'
    # torch.onnx.export(model, 
    #     inputs,
    #     output_onnx_name, 
    #     input_names=[  "x","zf",], 
    #     output_names=["output"],
    #     opset_version=11,
    #     export_params=True,
    #     # verbose=True,
    #     # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}} 
    # )
    # providers = ['CPUExecutionProvider']
    # onnx_model = onnx.load("test_net.onnx")
    
    # model_fp16 = float16.convert_float_to_float16(onnx_model)
    # model_fp16, success = simplify_func(onnx_model)
    # model_fp16 = float16.convert_float_to_float16(model_fp16)

    # onnx.save(model_fp16, "test_net_fp16.onnx")
    # ort_session = ort.InferenceSession("test_net_fp16.onnx", providers=providers)
    # # output = ort_session.run(output_names=['output'],
    # #                          	input_feed=inputs_onnx,
    # #                             )

    if use_gpu:
        model = model.cuda()
        x = x.cuda()
        zf = zf.cuda()
    # oup = model(x, zf)

    T_w = 100  # warmup
    T_t = 500  # test
    with torch.no_grad():
        for i in range(T_w):
            oup = model(x, zf)
            # output = ort_session.run(output_names=['output'],
            #                  	input_feed=inputs_onnx,
            #                     )
        t_s = time.time()
        for i in range(T_t):
            oup = model(x, zf)
            # output = ort_session.run(output_names=['output'],
            #                  	input_feed=inputs_onnx,
            #                     )
        t_e = time.time()
        print('speed: %.2f FPS' % (T_t / (t_e - t_s)))
