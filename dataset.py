# -*- coding: utf-8 -*-
import glob
import os
import random
import torch
import torch.utils.data as dd
import cv2
from tqdm import tqdm
from torchvision.transforms import ToTensor


"""
data on RAM, at least 20GB RAM is needed
all data is used to train model, there is no val set.
manually choose best model
"""


class TempData(dd.Dataset):
    def __init__(self, data, c=None):
        super().__init__()
        self.data = data
        self.c = c

    def __getitem__(self, item):
        if self.c == 'sam_train':
            im1 = self.data[item][0]
            im2 = self.data[item][1]
            im1 = ToTensor()(im1.copy())
            im2 = ToTensor()(im2.copy())
            label = self.data[item][2]
            label = label[:, :, 0]    # 0-1
            return im1, im2, label, self.data[item][3]

        if self.c == 'sam_test':
            im1 = self.data[item][0]
            im2 = self.data[item][1]
            im1 = ToTensor()(im1.copy())
            im2 = ToTensor()(im2.copy())
            return im1, im2, self.data[item][2]

        if self.c == 'train':
            im1 = self.data[item][0]
            im2 = self.data[item][1]
            im1 = ToTensor()(im1.copy())
            im2 = ToTensor()(im2.copy())
            label = self.data[item][2]
            label = label[:, :, 0]  # 0-1
            return im1, im2, label, self.data[item][3], self.data[item][4]

        if self.c == 'test':
            im1 = self.data[item][0]
            im2 = self.data[item][1]
            im1 = ToTensor()(im1.copy())
            im2 = ToTensor()(im2.copy())
            return im1, im2, self.data[item][2], self.data[item][3]

    def __len__(self):
        return len(self.data)


class Mydata(object):
    def __init__(self, data_root_dir='', c=''):
        self.data_root_dir = data_root_dir
        self.c = c
        self.dataset = None
        self.batch_size = 1
        self._set_dataset()

    def _set_dataset(self):
        self.dataset = []
        if self.c == 'sam_train' or self.c == 'train':
            im_names = [_.split('.')[0] for _ in os.listdir(os.path.join(self.data_root_dir, 'train', 'Image1'))]
        if self.c == 'sam_test' or self.c == 'test':
            im_names = [_.split('.')[0] for _ in os.listdir(os.path.join(self.data_root_dir, 'test', 'Image1'))]

        for name in tqdm(im_names, desc='load data: '):
            re = self.read_pair(name)
            re.append(name)
            self.dataset.append(re)

    def get_loader(self):
        sample_loader = dd.DataLoader(
            dataset=TempData(self.dataset, self.c),
            batch_size=self.batch_size,
            shuffle=True
        )

        return sample_loader

    def read_pair(self, im_name: str):
        """
        train: im1 im2 label
        va;: im1 im2
        """
        numpy_im = []
        format_ = {'Image1': '.tif', 'Image2': '.tif', 'label1': '.png', 'sam_feature': '.pth'}
        if self.c == 'sam_train':
            for s in ['Image1', 'Image2', 'label1']:
                numpy_im.append(cv2.imread(os.path.join(self.data_root_dir, 'train', s, im_name + format_[s])))
        if self.c == 'sam_test':
            for s in ['Image1', 'Image2']:
                numpy_im.append(cv2.imread(os.path.join(self.data_root_dir, 'test', s, im_name + format_[s])))
        if self.c == 'train':
            for s in ['Image1', 'Image2', 'label1']:
                numpy_im.append(cv2.imread(os.path.join(self.data_root_dir, self.c, s, im_name + format_[s])))
            numpy_im.append(torch.load(os.path.join(self.data_root_dir, self.c, 'sam_feature', im_name + '.pth')))
        if self.c == 'test':
            for s in ['Image1', 'Image2']:
                numpy_im.append(cv2.imread(os.path.join(self.data_root_dir, self.c, s, im_name + format_[s])))
            numpy_im.append(torch.load(os.path.join(self.data_root_dir, self.c, 'sam_feature', im_name + '.pth')))

        return numpy_im


if __name__ == '__main__':
    data = Mydata(data_root_dir='DATA/CD_dataset/', c='test')
    loader = data.get_loader()
    for im1, im2, sam_feature, name in tqdm(loader):
        print(im1.shape)
        print(im2.shape)
        print(sam_feature.shape)
        print(name[0])
    pass
