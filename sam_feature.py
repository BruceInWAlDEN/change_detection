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
from tqdm import tqdm


def make_sam_feature():
    """
    由于开源权重支持1024 * 1024 大小图片 需要将512大小图片进行拼接
    """

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

    # 提取所有训练集特征
    data = Mydata(data_root_dir='DATA/CD_dataset/', c='sam_train')
    data.batch_size = 4
    assert data.batch_size == 4, 'fix batch size == 4'
    loader = data.get_loader()

    os.mkdir('DATA/CD_dataset/train/sam_feature')

    for im1, im2, name in tqdm(loader, desc='make train sam feature'):

        # --> 4 3 512 512
        im1 = im1.to(device)
        im2 = im2.to(device)

        # --> 3 1024 1024
        ima = torch.cat([torch.cat([im1[0], im1[1]], dim=2), torch.cat([im1[2], im1[3]], dim=2)], dim=1)
        imb = torch.cat([torch.cat([im2[0], im2[1]], dim=2), torch.cat([im2[2], im2[3]], dim=2)], dim=1)

        # --> 2 3 1024 1024
        im = torch.stack([ima, imb], dim=0)

        with torch.no_grad():

            # --> 2 256 64 64
            feature = model(im)
            feature = feature.detach().clone().cpu()
            fa, fb = feature[0], feature[1]

            # --> 256 64 64 --> 256*2 32 32
            torch.save(torch.cat([fa[:, :32, :32], fb[:, :32, :32]], dim=0),
                       os.path.join('DATA/CD_dataset/train/sam_feature/{}.pth'.format(name[0])))
            torch.save(torch.cat([fa[:, :32, 32:], fb[:, :32, 32:]], dim=0),
                       os.path.join('DATA/CD_dataset/train/sam_feature/{}.pth'.format(name[1])))
            torch.save(torch.cat([fa[:, 32:, :32], fb[:, 32:, :32]], dim=0),
                       os.path.join('DATA/CD_dataset/train/sam_feature/{}.pth'.format(name[2])))
            torch.save(torch.cat([fa[:, 32:, 32:], fb[:, 32:, 32:]], dim=0),
                       os.path.join('DATA/CD_dataset/train/sam_feature/{}.pth'.format(name[3])))

    # 提取所测试集特征
    data = Mydata(data_root_dir='DATA/CD_dataset/', c='sam_test')
    data.batch_size = 4
    assert data.batch_size == 4, 'fix batch size == 4'
    loader = data.get_loader()

    os.mkdir('DATA/CD_dataset/test/sam_feature')

    for im1, im2, name in tqdm(loader, desc='make test sam feature'):
        # --> 4 3 512 512
        im1 = im1.to(device)
        im2 = im2.to(device)

        # --> 3 1024 1024
        ima = torch.cat([torch.cat([im1[0], im1[1]], dim=2), torch.cat([im1[2], im1[3]], dim=2)], dim=1)
        imb = torch.cat([torch.cat([im2[0], im2[1]], dim=2), torch.cat([im2[2], im2[3]], dim=2)], dim=1)

        # --> 2 3 1024 1024
        im = torch.stack([ima, imb], dim=0)

        with torch.no_grad():
            # --> 2 256 64 64
            feature = model(im)
            feature = feature.detach().clone().cpu()
            fa, fb = feature[0], feature[1]

            # --> 256 64 64 --> 256*2 32 32
            torch.save(torch.cat([fa[:, :32, :32], fb[:, :32, :32]], dim=0),
                       os.path.join('DATA/CD_dataset/test/sam_feature/{}.pth'.format(name[0])))
            torch.save(torch.cat([fa[:, :32, 32:], fb[:, :32, 32:]], dim=0),
                       os.path.join('DATA/CD_dataset/test/sam_feature/{}.pth'.format(name[1])))
            torch.save(torch.cat([fa[:, 32:, :32], fb[:, 32:, :32]], dim=0),
                       os.path.join('DATA/CD_dataset/test/sam_feature/{}.pth'.format(name[2])))
            torch.save(torch.cat([fa[:, 32:, 32:], fb[:, 32:, 32:]], dim=0),
                       os.path.join('DATA/CD_dataset/test/sam_feature/{}.pth'.format(name[3])))


if __name__ == '__main__':
    make_sam_feature()
