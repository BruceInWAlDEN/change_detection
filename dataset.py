# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.utils.data as dd
import cv2
from tqdm import tqdm
from torchvision.transforms import ToTensor

"""
将原始训练集划分为训练集和验证集
"""


def train_val(val_ratio=0.2):
    root_dir = '../CD_dataset/train'
    all_im = [_.split('.')[0] for _ in os.listdir(os.path.join(root_dir, 'Image1'))]
    random.shuffle(all_im)
    all_num = len(all_im)
    val_set = all_im[:int(all_num * val_ratio)]
    train_set = all_im[int(all_num * val_ratio):]

    torch.save(val_set, 'val_set.pth')
    torch.save(train_set, 'train_set.pth')


"""
data on RAM, at least 10GB RAM is needed
"""


class TempData(dd.Dataset):
    def __init__(self, data, c=None):
        super().__init__()
        self.data = data
        self.c = c

    def __getitem__(self, item):
        if self.c == 'train' or self.c == 'val':
            im1 = self.data[item][0]
            im2 = self.data[item][1]
            im1 = ToTensor()(im1.copy())
            im2 = ToTensor()(im2.copy())
            label = self.data[item][2]
            label = label[:, :, 0]    # 0-1
            return im1, im2, label

        if self.c == 'test':
            im1 = self.data[item][0][0]
            im2 = self.data[item][0][1]
            im1 = ToTensor()(im1.copy())
            im2 = ToTensor()(im2.copy())
            return im1, im2, self.data[item][1]

    def __len__(self):
        return len(self.data)


class Mydata(object):
    def __init__(self, c=None):
        self.data = torch.load('train_set.pth' if c == 'train' else 'val_set.pth')
        self.dataset = None     # [[iamg1, iamge2, label], []]
        self.batch_size = None
        self._set_dataset()

    def _set_dataset(self):
        self.dataset = []
        for im in tqdm(self.data, desc='LOAD DATA: '):
            self.dataset.append(self.read_pair(im))

    def get_loader(self):
        sample_loader = dd.DataLoader(
            dataset=TempData(self.dataset),
            batch_size=self.batch_size,
            shuffle=True
        )

        return sample_loader

    @staticmethod
    def read_pair(im_name: str):
        root_dir = '../CD_dataset/train'
        numpy_im = []
        format_ = {'Image1': '.tif', 'Image2': '.tif', 'label1': '.png'}
        for s in ['Image1', 'Image2', 'label1']:
            numpy_im.append(cv2.imread(os.path.join(root_dir, s, im_name + format_[s])))

        return numpy_im


class Testdata(object):
    def __init__(self, test_dir=None):
        if test_dir is None:
            self.test_dir = '../CD_dataset/test'
            self.data = [_.split('.')[0] for _ in os.listdir('../CD_dataset/test/Image1')]
        else:
            self.test_dir = test_dir
            self.data = [_.split('.')[0] for _ in os.listdir(os.path.join(test_dir, 'Image1'))]
        self.dataset = None
        self._set_dataset()

    def _set_dataset(self):
        self.dataset = []
        for im in tqdm(self.data, desc='LOAD DATA: '):
            self.dataset.append((self.read_pair(im), im))

    def get_loader(self):
        sample_loader = dd.DataLoader(
            dataset=TempData(self.dataset, 'test'),
            batch_size=1,
            shuffle=True
        )

        return sample_loader

    def read_pair(self, im_name: str):
        root_dir = self.test_dir
        numpy_im = []
        format_ = {'Image1': '.tif', 'Image2': '.tif'}
        for s in ['Image1', 'Image2']:
            numpy_im.append(cv2.imread(os.path.join(root_dir, s, im_name + format_[s])))

        return numpy_im


if __name__ == '__main__':
    pass
