# -*- coding: utf-8 -*-
"""
make sam feature
input: [{
    'image': tensor 0-255 RGB CHW
    'original_size': (H, W)
    'point_labels': B N 2
    'point_labels': B N
    'boxes': B 4
    'mask_inputs': B 1 H W
}, ...]
out_put: [{
    'masks': B(input prompts num) C(multi task) H W
}, ...]
"""
from segment_anything import build_sam_vit_b
from segment_anything.modeling.image_encoder import ImageEncoderViT
import torch
import os
import glob


def make_data():
    pass


def make_sam_feature():
    weight = torch.load('sam_vit_b_01ec64.pth')
    weight = {k.replace('image_encoder.', ''): v for k, v in weight.items() if 'image_encoder.' in k}
    model = ImageEncoderViT(img_size=512)
    model_weight = model.state_dict()
    model_weight.update(weight)
    model.load_state_dict(model_weight)
    model.eval()

    pass
    #
    # img = torch.rand(3, 3, 512, 512)
    # print(img.shape)
    # out = model(img)
    # print(out.shape)
    """
    torch.Size([3, 3, 512, 512])
    torch.Size([3, 256, 32, 32])
    """


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
    #
    # for k in m_key:
    #     if k not in d_key:
    #         print(k)
    make_sam_feature()
    pass