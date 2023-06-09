# -*- coding: utf-8 -*-
import glob

import numpy as np

from dataset import Testdata, Mydata
import torch
from model import Changer, base
from tqdm import tqdm
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

test_v1 = {
    'cuda_id': 1,
    'test_dir': '../CD_dataset/test',
    'result_save_dir': 'submit',
    'model_weight': 'log/CD_v1_log/CD_v1_97.pth'
}


def test_pair(im1, im2, model):
    """
    :param im1: tensor C H W   cuda
    :param im2: tensor C H W   cuda
    :param model: cuda
    :return: mask
    """
    im1 = im1.unsqueeze(0)
    im2 = im2.unsqueeze(0)

    with torch.no_grad():
        outputs = model(im1, im2)

    return outputs.squeeze()


def test(cfg):
    device = torch.device('cuda', index=0)

    # test_data = Testdata(cfg['test_dir'])
    # loader = test_data.get_loader()
    test_data = Testdata()
    test_data.batch_size = 1
    loader = test_data.get_loader()
    model = Changer(**base)
    w = torch.load(cfg['model_weight'], map_location='cpu')['model_weights']
    model.load_state_dict(w)
    model.to(device)
    model.eval()

    result = []
    for im1, im2, name in tqdm(loader, desc='Reference: '):
        im1 = im1.to(device)
        im2 = im2.to(device)
        im1 = im1.squeeze(0)
        im2 = im2.squeeze(0)
        mask = test_pair(im1, im2, model)
        # submit.append(mask)
        mask = mask.detach().clone().cpu()

        pre = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask)).long()
        result.append((np.uint8(pre.numpy()), name))
        # plt.subplot(1,2,1)
        # plt.imshow(im1.cpu().numpy().transpose(1,2,0))
        # plt.subplot(1,2,2)
        # plt.imshow(im2.cpu().numpy().transpose(1,2,0))
        # plt.show()
        #
        # plt.matshow(np.concatenate([label.squeeze().numpy(), pred], axis=1))
        # plt.show()
        # break

    for pre, name in result:
        img = Image.fromarray(pre, mode='L')
        img.save('./submit/' + name[0] + '.png')


if __name__ == '__main__':
    test(test_v1)
