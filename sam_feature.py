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
from model.segment_anything.modeling.image_encoder import ImageEncoderViT
import torch
from dataset import Mydata
import os


def make_sam_feature():

    device = torch.device('cuda', index=0)

    weight = torch.load('DATA/sam_vit_b_01ec64.pth')
    weight = {k.replace('image_encoder.', ''): v for k, v in weight.items() if 'image_encoder.' in k}
    weight_k = list(weight.keys())
    model = ImageEncoderViT(img_size=1024)
    model_weight = model.state_dict()

    for k in model_weight.keys():
        if k in weight_k:
            model_weight[k] = weight[k]
        else:
            print('lost weight: ', k)

    model.load_state_dict(model_weight)
    model.eval()
    model.to(device)

    test_in = torch.rand(4, 3, 1024, 1024)
    with torch.no_grad():
        out = model(test_in.to(device))
    print(out.shape)
    raise AssertionError
    """
    input: torch.Size([3, 3, 512, 512])
    output: torch.Size([3, 256, 32, 32])
    """

    # 提取所有训练集特征
    data = Mydata(data_root_dir='DATA/CD_dataset/', c='train')
    data.batch_size = 1
    loader = data.get_loader()

    os.mkdir('DATA/CD_dataset/train/sam_feature')

    for im1, im2, label, name in loader:
        im1 = im1.to(device)
        im2 = im2.to(device)

        batch_size = len(name)
        # --> B*2 3 512 512
        im = torch.cat([im1, im2], dim=0)

        with torch.no_grad():

            # --> B*2 256 32 32
            feature = model(im)
            out = feature.detach().clone().cpu()

        for index in range(len(name)):

            # --> 256*2 32 32
            torch.save(torch.cat([out[index], out[index+batch_size]], dim=0),
                       os.path.join('DATA/CD_dataset/train/sam_feature/{}.pth'.format(name[index])))

    # 提取所测试集特征
    data = Mydata(data_root_dir='DATA/CD_dataset/', c='test')
    data.batch_size = 4
    loader = data.get_loader()

    os.mkdir('DATA/CD_dataset/test/sam_feature')

    for im1, im2, label, name in loader:
        im1 = im1.to(device)
        im2 = im2.to(device)

        batch_size = len(name)
        # --> B*2 3 512 512
        im = torch.cat([im1, im2], dim=0)

        with torch.no_grad():

            # --> B*2 256 32 32
            feature = model(im)
            out = feature.detach().clone().cpu()

        for index in range(len(name)):

            # --> 256*2 32 32
            torch.save(torch.cat([feature[index], feature[index+batch_size]], dim=0),
                       os.path.join('DATA/CD_dataset/test/sam_feature/{}.pth'.format(name[index])))
    pass


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