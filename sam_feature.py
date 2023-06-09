# -*- coding: utf-8 -*-

import os, sys

sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores, boxes


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# r_model_path = "/home/zigangzhao/DMS/mtcnn-pytorch/test0815/onnx_model/rnet.onnx"
# o_model_path = "/home/zigangzhao/DMS/mtcnn-pytorch/test0815/onnx_model/onet.onnx"
#
# img = cv2.imread("/home/zigangzhao/DMS/mtcnn-pytorch/data_set/train/24/positive/999.jpg")
# img = cv2.resize(img, 24, 24), interpolation = cv2.INTER_CUBIC)
#
# """
# # scipy.misc.imread 读取的图片数据是 RGB 格式
# # cv2.imread 读取的图片数据是 BGR 格式
# # PIL.Image.open 读取的图片数据是RGB格式
# # 注意要与pth测试时图片读入格式一致
# """
# to_tensor = transforms.ToTensor()
# img = to_tensor(img)
# img = img.unsqueeze_(0)
#
#
# rnet1 = ONNXModel(r_model_path)
# out = rnet1.forward(to_numpy(img))
# print(out)
# rnet_session = onnxruntime.InferenceSession(r_model_path)
# onet_session = onnxruntime.InferenceSession(o_model_path)
# # compute ONNX Runtime output prediction
# inputs = {onet_session.get_inputs()[0].name: to_numpy(img)}
# outs = onet_session.run(None, inputs)
# img_out_y = outs
#
# print(img_out_y)


# def build_sam_vit_h(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=1280,
#         encoder_depth=32,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[7, 15, 23, 31],
#         checkpoint=checkpoint,
#     )
#
#
# def build_sam_vit_l(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=1024,
#         encoder_depth=24,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[5, 11, 17, 23],
#         checkpoint=checkpoint,
#     )
#
#
# def build_sam_vit_b(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=768,
#         encoder_depth=12,
#         encoder_num_heads=12,
#         encoder_global_attn_indexes=[2, 5, 8, 11],
#         checkpoint=checkpoint,
#     )


if __name__ == '__main__':
    # from modeling.image_encoder import ImageEncoderViT
    # model = ImageEncoderViT(        # image size 1024
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     global_attn_indexes=(2, 5, 8, 11),
    # )
    # d = torch.load('sam_vit_b_01ec64.pth')
    # d_key = [k.replace('image_encoder.', '') for k in d.keys() if 'image_encoder' in k]
    # print(len(d_key))
    # m_key = [_ for _ in model.state_dict().keys()]
    # print(len(m_key))
    #
    # d_key.sort()
    # m_key.sort()
    # print(d_key)
    # print(m_key)

    # for k in m_key:
    #     if k not in d_key:
    #         print(k)
    from segment_anything import build_sam_vit_b
    model = build_sam_vit_b(checkpoint='sam_vit_b_01ec64.pth')




