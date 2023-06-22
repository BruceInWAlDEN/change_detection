# -*- coding: utf-8 -*-
import glob

import numpy as np
import torch.nn.functional as F
from dataset import Mydata
import torch
from model.model import MixChanger, MixChanger_base
from tqdm import tqdm
from PIL import Image
import os
import cv2
import pdb
import matplotlib.pyplot as plt
import pdb

test_cfg = {
    'cuda_id': 1,
    'result_save_dir': 'submit',
    'model_weight': 'DATA/MixChanger_v2_log/MixChanger_v2_121.pth',
    'batch_size': 4
}


def mIOU(predict, label):
    """
    predict: B 2 H W  tensor cpu
    label: B 1 H W  0-1 tensor cpu
    """
    # --> B*H*W 2
    predict = predict.permute(0, 2, 3, 1).reshape(-1, 2)
    # --> B*H*W
    label = label.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
    pre = torch.argmax(F.softmax(predict, dim=1), dim=1)
    TP, FP, TN, FN = 0, 0, 0, 0
    for index in range(label.shape[0]):
        if pre[index] == 1 and label[index] == 1:
            TP += 1
        if pre[index] == 0 and label[index] == 0:
            TN += 1
        if pre[index] == 1 and label[index] == 0:
            FP += 1
        if pre[index] == 0 and label[index] == 1:
            FN += 1

    score = 0.5 * TP / (TP + FP + FN) + 0.5 * TN / (TN + FP + FN)

    return score, TP, FP, TN, FN


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
    test_data = Mydata(data_root_dir='DATA/CD_dataset', c='val')
    test_data.batch_size = cfg['batch_size']
    test_data.set_dataset(1000)
    test_loader = test_data.get_loader()

    # model
    w = torch.load(cfg['model_weight'], map_location='cpu')['model_weights']
    model.load_state_dict(w)
    model.to(device)
    model.eval()

    result = []
    for im1, im2, label, name in tqdm(test_loader, desc='Reference: '):
        im1 = im1.to(device)
        im2 = im2.to(device)

        with torch.no_grad():
            mask = model(im1, im2)

        mask = mask.detach().clone().cpu()

        result.append(mIOU(mask, label))
        # for index in range(len(name)):
        #     pre = torch.argmax(mask[index], dim=0)
        #     label_ = label[index].squeeze()
        #     plt.subplot(1,2,1)
        #     plt.imshow(im1[index].cpu().numpy().transpose(1,2,0))
        #     plt.subplot(1,2,2)
        #     plt.imshow(im2[index].cpu().numpy().transpose(1,2,0))
        #     plt.show()
        #
        #     plt.matshow(np.concatenate([label_.numpy(), pre.numpy()], axis=1)/1.)
        #     plt.show()
        #
    score = [_[0] for _ in result]
    print('score', sum(score)/len(score))


if __name__ == '__main__':
    show_test(test_cfg)
