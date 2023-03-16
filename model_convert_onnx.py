import os
import onnxruntime
import torch
import numpy as np
import onnx
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from onnxsim import simplify

from models.crnn.crnn_models import CRNN
from models.efficientnet.model import EfficientNet


def softmax_2D(X):
    """
    针对二维numpy矩阵每一行进行softmax操作
    X: np.array. Probably should be floats.
    return: 二维矩阵
    """
    # looping through rows of X
    #   循环遍历X的行
    ps = np.empty(X.shape)
    for i in range(X.shape[0]):
        ps[i, :] = np.exp(X[i, :])
        ps[i, :] /= np.sum(ps[i, :])
    return ps


def model_convert_onnx(model, input_shape, output_path, device):
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
    dummy_input = dummy_input.to(device)
    # 输入的名称和输出的名称
    input_names = ["input1"]
    output_names = ["output1"]

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        verbose=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,  # 是否执行常量折叠优化
        opset_version=11,  # 版本通常为10 or 11
        input_names=input_names,
        output_names=output_names,
    )


def check_onnx_2(model, ort_session, input_shape, device):
    x = torch.randn(size=(1, 3, input_shape[0], input_shape[1]), dtype=torch.float32)
    # torch模型推理
    with torch.no_grad():
        torch_out = model(x.to(device))
    # print(torch_out)
    # print(type(torch_out))      # <class 'torch.Tensor'>
    # onnx模型推理
    # 初始化数据，注意这儿的x是上面的输入数据x，后期应该是img
    ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
    # print(type(ort_outs))       # <class 'list'>，里面是个numpy矩阵
    # print(type(ort_outs[0]))    # <class 'numpy.ndarray'>
    # ort_outs = ort_outs[0]  # 因此这儿需要把内部numpy矩阵取出来，这一步很有必要
    # np.testing.assert_allclose(torch_out.cpu().numpy(), ort_outs, rtol=1e-03, atol=1e-05)


def check_onnx_3(ort_session, img, input_shape):
    img = img.convert('RGB')
    img_resize = img.resize(input_shape, Image.BICUBIC)  # PIL.Image类型
    # PIL.Image类型无法直接除以255，需要先转成array
    img_resize = np.array(img_resize, dtype='float32') / 255.0
    img_resize -= [0.485, 0.456, 0.406]
    img_resize /= [0.229, 0.224, 0.225]
    img_CHW = np.transpose(img_resize, (2, 1, 0))
    img = np.expand_dims(img_CHW, 0)
    # onnx模型推理
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)  # 推理得到输出
    print(ort_outs)
    # # 分类的名称映射
    # class_indict = {"1": "dog", "0": "cat"}
    # predict_probability = softmax_2D(ort_outs[0])
    # predict_cla = np.argmax(predict_probability, axis=-1)
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla[0])],
    #                                              predict_probability[0][predict_cla[0]])
    # plt.title(print_res)
    # for i in range(len(predict_probability[0])):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict_probability[0][i]))
    # plt.show()


def model_sim(output_path):
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    sim_name = output_path.replace(".onnx", "_sim.onnx")
    onnx.save(model_simp, sim_name)
    print('finished exporting onnx')


def convert_PT2ONNX(model, device, output_path):
    # 导出onnx模型的输入尺寸，要和pytorch模型的输入尺寸一致
    # onnx的输入可以是动态的，请自行查资料，这里先固定
    input_shape = (32, 480)
    device = torch.device(device)
    # 进行onnx转化
    model_convert_onnx(model, input_shape, output_path, device)
    print("model convert onnx finsh.")


def get_pt_model(device):
    # 创建模型
    model = CRNN(img_h=32, nc=3, n_class=9464, nh=256)
    checkpoint = torch.load("weights/crnn.pth", device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def check_onnx_model(model, device, output_path):
    # -------------------------#
    input_shape = (32, 480)
    #  第一轮验证
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("第一轮验证：onnx model check_1 finsh.")
    # -------------------------#
    # 第二轮验证
    # 初始化onnx模型
    ort_session_1 = onnxruntime.InferenceSession(output_path)
    check_onnx_2(model, ort_session_1, input_shape, device)
    print("onnx model check_2 finsh.")
    # -------------------------#
    #   第三轮验证
    img_path = "./data/single_line.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # 加载onnx模型
    ort_session_2 = onnxruntime.InferenceSession(output_path)
    check_onnx_3(ort_session_2, img, input_shape)
    print("onnx model check_3 finsh.")
    # -------------------------#


if __name__ == '__main__':
    device = "cpu"
    # onnx模型输出保存路径
    output_path = './output/efficientnet.onnx'
    model = get_pt_model(device)
    # print(model)
    # onnx_model = convert_PT2ONNX(model, device, output_path)
    check_onnx_model(model, device, output_path)
    #   进行模型精简
    model_sim(output_path)
