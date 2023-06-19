# -*- coding: utf-8 -*-
import glob

import numpy as np

from dataset import Mydata
import torch
from model.model import MixChanger, MixChanger_base
from tqdm import tqdm
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

test_v2 = {
    'cuda_id': 1,
    'result_save_dir': 'submit',
    'model_weight': 'DATA/MixChanger_v4_353.pth',
    'batch_size': 4
}


def test(cfg):
    device = torch.device('cuda', index=0)

    # model
    model = MixChanger(**MixChanger_base)
    model.to(device)

    # data
    test_data = Mydata(data_root_dir='DATA/CD_dataset', c='test')
    test_data.batch_size = cfg['batch_size']
    test_loader = test_data.get_loader()

    # model
    w = torch.load(cfg['model_weight'], map_location='cpu')['model_weights']
    model.load_state_dict(w)
    model.to(device)
    model.eval()

    result = []
    for im1, im2, sam_feature, name in tqdm(test_loader, desc='Reference: '):
        im1 = im1.to(device)
        im2 = im2.to(device)
        sam_feature = sam_feature.to(device)

        with torch.no_grad():
            mask = model(im1, im2, sam_feature)

        mask = mask.detach().clone().cpu()

        for index in range(len(name)):
            pre = torch.where(mask[index] > 0.5, torch.ones_like(mask[index]), torch.zeros_like(mask[index])).long()
            result.append((np.uint8(pre.numpy()), name[index]))

            # plt.subplot(1,2,1)
            # plt.imshow(im1[index].cpu().numpy().transpose(1,2,0))
            # plt.subplot(1,2,2)
            # plt.imshow(im2[index].cpu().numpy().transpose(1,2,0))
            # plt.show()
            #
            # plt.matshow(np.concatenate([label[index].squeeze().numpy(), pre], axis=1))
            # plt.show()

    # save
    for mask, name in result:
        img = Image.fromarray(mask, mode='L')  # 创建PIL图像对象
        img.save("DATA/submit/" + name + '.png')


def show_test(cfg):
    device = torch.device('cuda', index=0)

    # model
    model = MixChanger(**MixChanger_base)
    model.to(device)

    # data
    test_data = Mydata(data_root_dir='DATA/CD_dataset', c='train')
    test_data.batch_size = cfg['batch_size']
    test_loader = test_data.get_loader()

    # model
    w = torch.load(cfg['model_weight'], map_location='cpu')['model_weights']
    model.load_state_dict(w)
    model.to(device)
    model.eval()

    result = []
    for im1, im2, label, sam_feature, name in tqdm(test_loader, desc='Reference: '):
        im1 = im1.to(device)
        im2 = im2.to(device)
        sam_feature = sam_feature.to(device)

        with torch.no_grad():
            mask = model(im1, im2, sam_feature)

        mask = mask.detach().clone().cpu()

        for index in range(len(name)):
            pre = torch.where(mask[index] > 0.5, torch.ones_like(mask[index]), torch.zeros_like(mask[index])).long()
            result.append((np.uint8(pre.numpy()), name[index]))

            plt.subplot(1,2,1)
            plt.imshow(im1[index].cpu().numpy().transpose(1,2,0))
            plt.subplot(1,2,2)
            plt.imshow(im2[index].cpu().numpy().transpose(1,2,0))
            plt.show()

            plt.matshow(np.concatenate([label[index].squeeze().numpy(), pre], axis=1))
            plt.show()


if __name__ == '__main__':
    show_test(test_v2)
