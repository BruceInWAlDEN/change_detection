# -*- coding: utf-8 -*-
"""
复现一下changer先
paper: Changer: Feature Interaction is What You Need for Change Detection
ori code: 写在open-cd中，LEVIR数据集最好的模型是ChangerEx
    # transformer 作为对序列进行处理的结构，本身没有Channel的概念。根据changer提出的在空间和通道的维度进行交换，
        若使用transforme结构，需要将序列形式的特征reshape为CWH形式，但代表patch的token其本身并没有通道这一概念
        若只是对token进行交换，虽然类似空间交换，但是这种空间交换以patch宽度来交换的。
    # 图像的差值和图像本身不适合用同一网络来处理，但是图像的差值又是很好的特征形式，同时单输入图像的差值不好

"""
import torch.nn as nn
import torch
from transformer_layers import Block
from changer_layers import FDAF, MixFFN, SpatialExchange, ChannelExchange


base = {
    # encoder hyper
    'patch_size': 16,
    'channel': 3,
    'dim': 768,
    'num_head': 12,
    'mlp_ratio': 4,
}


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.mini_cnn = nn.Sequential(
            nn.Conv2d(3, 3*4, (3, 3), padding='same', padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(3*4, 3, (3, 3), padding='same',  padding_mode='reflect'),
            nn.GELU(),
        )
        self.mini_transformer = nn.Sequential(
            Block(dim=768, num_head=12, mlp_ratio=4),
            Block(dim=768, num_head=12, mlp_ratio=4),
        )
        self.post = nn.Sequential(
            nn.Conv2d(6, 1, (3, 3), padding='same',  padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入图像对的差值
        """
        x1 = self.mini_cnn(x)

        x2 = image2squence(x, patch_size=16)
        x2 = self.mini_transformer(x2)
        x2 = squence2image(x2, patch_size=16, channel=3)

        x = torch.cat([x1, x2], dim=1)
        x = self.post(x)

        return x


"""
选取transformer作为backbone时，channel对应embedding_dim,将patch_num 拆分为W H维度
"""


class Changer(nn.Module):
    def __init__(self, dim, channel, num_head, mlp_ratio, patch_size, depth=None):
        super().__init__()
        if depth is None:
            depth = [3, 3, 3, 3]
        self.patch_size = patch_size
        self.channel = channel
        self.depth = depth
        self.dim = dim
        self.project = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel * 2, kernel_size=(3, 3), padding='same', padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(self.channel * 2, 1, kernel_size=(1, 1), padding='same', padding_mode='reflect')
        )
        self.encoder_1 = nn.ModuleList([
            Block(dim=dim, num_head=num_head, mlp_ratio=mlp_ratio
            ) for _ in range(self.depth[0])
        ])
        self.encoder_2 = nn.ModuleList([
            Block(dim=dim, num_head=num_head, mlp_ratio=mlp_ratio
            ) for _ in range(self.depth[0])
        ])
        self.encoder_3 = nn.ModuleList([
            Block(dim=dim, num_head=num_head, mlp_ratio=mlp_ratio
            ) for _ in range(self.depth[0])
        ])
        self.encoder_4 = nn.ModuleList([
            Block(dim=dim, num_head=num_head, mlp_ratio=mlp_ratio
            ) for _ in range(self.depth[0])
        ])
        self.spatial_exchange = SpatialExchange()
        self.channel_exchange = ChannelExchange()

    def forward(self, x1, x2):
        # preprocess
        x1 = image2squence(x1, self.patch_size) # B patch_num patch**2 * 3
        x2 = image2squence(x2, self.patch_size) # B patch_num patch**2 * 3

        # B patch_num dim -> B patch_num dim
        for blk in self.encoder_1:
            x1 = blk(x1)
            x2 = blk(x2)

        # B patch_num dim -> B patch_num dim
        for blk in self.encoder_2:
            x1 = blk(x1)
            x2 = blk(x2)

        # B patch_num dim -> B patch_num dim
        x1 = squence2image(x1, self.patch_size, self.channel)
        x2 = squence2image(x2, self.patch_size, self.channel)
        x1, x2 = self.spatial_exchange(x1, x2)
        x1 = image2squence(x1, self.patch_size)
        x2 = image2squence(x2, self.patch_size)

        # B patch_num dim -> B patch_num dim
        for blk in self.encoder_3:
            x1 = blk(x1)
            x2 = blk(x2)

        # B patch_num dim -> B patch_num dim
        x1 = squence2image(x1, self.patch_size, self.channel)
        x2 = squence2image(x2, self.patch_size, self.channel)
        x1, x2 = self.channel_exchange(x1, x2)
        x1 = image2squence(x1, self.patch_size)
        x2 = image2squence(x2, self.patch_size)

        # B patch_num dim -> B patch_num dim
        for blk in self.encoder_4:
            x1 = blk(x1)
            x2 = blk(x2)

        # B patch_num dim -> B patch_num dim
        x1 = squence2image(x1, self.patch_size, self.channel)
        x2 = squence2image(x2, self.patch_size, self.channel)
        x1, x2 = self.channel_exchange(x1, x2)

        # B C W H
        mask = self.project(torch.cat([x1, x2], dim=1))

        return mask


def image2squence(tensor_im, patch_size):
    """
    input: N C H W   tensor cuda RGB
    output: N patch_num dim  cuda tensor
    图像到patch的顺序为从左到右，从上到下。
    patch到vector的顺序为像素从左到右，从上到小,从R到B  tensor.flatten()
    """
    temp = []
    im_size = tensor_im.shape[-1]
    patch_num_ = im_size // patch_size
    for hh in range(patch_num_):
        for ww in range(patch_num_):
            temp.append(tensor_im[:, :, hh*patch_size:hh*patch_size+patch_size, ww*patch_size:ww*patch_size+patch_size])
    temp = torch.stack(temp, dim=1)
    temp = temp.flatten(2)

    return temp


def squence2image(tensor, patch_size, channel):
    """
    input: N patch_num dim  cuda tensor
    output: N C H W   cuda tensor
    恢复顺序按照image2squence的相反顺序恢复  tensor.reshape(3, patch_size, patch_size)
    """
    N, patch_num = tensor.shape[:2]
    tensor = tensor.reshape(N, patch_num, channel, patch_size, patch_size)
    patch_num_ = int(patch_num**0.5)
    tensor = [tensor[:, _, :, :, :] for _ in range(patch_num)]
    tensor = [torch.cat(tensor[_*patch_num_:_*patch_num_+patch_num_], dim=3) for _ in range(patch_num_)]
    tensor = torch.cat(tensor, dim=2)

    return tensor


if __name__ == '__main__':
    im = torch.rand(3, 3, 512, 512)
    model = Changer(**base)
    out = model(im, im)
    print(out.shape)