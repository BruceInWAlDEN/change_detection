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
import torch.nn.functional as F


changer_base = {
    # encoder hyper
    'patch_size': 16,
    'channel': 3,
    'dim': 768,
    'num_head': 12,
    'mlp_ratio': 4,
}


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


MixChanger_base = {
    'encoder_dim': 768,
    'input_channel': 3,
    'patch_size': 16,
    'encoder_num_head': 12,
    'encoder_mlp_ratio': 4,
    'depth': None,
    'sam_feature_dim': 512
}


class MixChanger(nn.Module):
    def __init__(self,
                 encoder_dim=768,
                 input_channel=3,
                 patch_size=16,
                 encoder_num_head=12,
                 encoder_mlp_ratio=4,
                 depth=None,
                 sam_feature_dim=512
                 ):
        super().__init__()
        if depth is None:
            depth = [2, 2, 2, 2]
        self.sam_feature_dim = sam_feature_dim
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.input_channel = input_channel
        self.depth = depth
        self.patch_embedding = nn.Conv2d(
            in_channels=self.input_channel,
            out_channels=self.encoder_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )
        self.encoder_stage_one = nn.ModuleList([
            Block(dim=encoder_dim, num_head=encoder_num_head, mlp_ratio=encoder_mlp_ratio
            ) for _ in range(depth[0])
        ])
        self.encoder_stage_two = nn.ModuleList([
            Block(dim=encoder_dim, num_head=encoder_num_head, mlp_ratio=encoder_mlp_ratio
            ) for _ in range(depth[1])
        ])
        self.encoder_stage_three = nn.ModuleList([
            Block(dim=encoder_dim, num_head=encoder_num_head, mlp_ratio=encoder_mlp_ratio
            ) for _ in range(depth[2])
        ])
        self.encoder_stage_four = nn.ModuleList([
            Block(dim=encoder_dim, num_head=encoder_num_head, mlp_ratio=encoder_mlp_ratio
            ) for _ in range(depth[3])
        ])
        self.sam_encoder = nn.Conv2d(
            in_channels=self.sam_feature_dim,
            out_channels=self.encoder_dim,
            kernel_size=(1, 1)
        )
        self.attention_feature_merge = nn.Linear(self.encoder_dim, 1)
        self.MixFFN = MixFFN(6 * self.input_channel, 6 * self.input_channel)

        self.ffn_and_expand = nn.Sequential(
            nn.Linear(self.encoder_dim * 2, int(patch_size * patch_size * self.input_channel)),
            nn.GELU(),
            nn.Linear(int(patch_size * patch_size * self.input_channel), int(patch_size * patch_size * self.input_channel))
        )
        self.attention_im_merge = nn.Linear(encoder_dim, 1)

        self.ffn_and_project = nn.Conv2d(
            in_channels=6 * self.input_channel,
            out_channels=1,
            kernel_size=(1, 1)
        )

        self.spatial_exchange = SpatialExchange()
        self.channel_exchange = ChannelExchange()

    def forward(self, im1, im2, sam_feature):
        """
        im1. im2: (B 3 512 512)  0-1 RGB
        sam_feature: (B 512 32 32)
        --> mask (B im_h im_w)
        """
        # B 3 512 512 --> B dim 32 32
        xa1 = self.patch_embedding(im1)
        xb1 = self.patch_embedding(im2)

        # B dim 32 32 --> B dim 32 32
        for blk in self.encoder_stage_one:
            xa1 = blk(xa1)
            xb1 = blk(xb1)

        xa2, xb2 = xa1, xb1

        # B dim 32 32 --> B dim 32 32
        for blk in self.encoder_stage_two:
            xa2 = blk(xa2)
            xb2 = blk(xb2)

        xa3, xb3 = xa2, xb2
        xa3, xb3 = self.spatial_exchange(xa3, xb3)

        # B dim 32 32 --> B dim 32 32
        for blk in self.encoder_stage_three:
            xa3 = blk(xa3)
            xb3 = blk(xb3)

        xa4, xb4 = xa3, xb3
        xa4, xb4 = self.channel_exchange(xa4, xb4)

        # B dim 32 32 --> B dim 32 32
        for blk in self.encoder_stage_four:
            xa4 = blk(xa4)
            xb4 = blk(xb4)

        xa5, xb5 = xa4, xb4
        xa5, xb5 = self.channel_exchange(xa5, xb5)

        # B sam_feature_dim 32 32 --> B dim 32 32
        sam_feature = self.sam_encoder(sam_feature)

        # --> 10 B dim 32 32
        feature_merge = torch.stack([xa5, xa4, xa3, xa2, xa1, xb5, xb4, xb3, xb2, xb1], dim=0)

        # --> B 32 32 10 dim --> B 32 32 10
        attention_score = self.attention_feature_merge(feature_merge.permute(1, 3, 4, 0, 2))
        attention_score = F.softmax(attention_score.squeeze(), dim=3)

        # --> B 32 32 dim
        feature_merge = feature_merge.permute(1, 3, 4, 2, 0) @ attention_score.unsqueeze(0).permute(1, 2, 3, 4, 0)
        feature_merge = feature_merge.squeeze()

        # --> B 32 32 dim --> B 32 32 dim*2
        feature_merge = torch.cat([feature_merge, sam_feature], dim=3)

        # B 32 32 dim --> B 32 32 patch*patch*3 --> B patch*patch*3 32 32
        expand_feature = self.ffn_and_expand(feature_merge).permute(0, 3, 1, 2)

        # --> B patch*patch*input_channel 32 32
        # --> B 32 32 patch*patch*input_channel --> B 32 32 input_channel patch_size patch_size
        batch, dim, patch_h, patch_w = expand_feature.shape
        expand_feature = expand_feature.permute(0, 2, 3, 1).reshape(batch, patch_h, patch_w, self.input_channel, self.patch_size, self.patch_size)

        # [先从左到右 后从上到下] // B input_channel patch_size patch_size
        expand_feature = [expand_feature[:, hh, ww, :, :, :] for hh in range(patch_h) for ww in range(patch_w)]
        expand_feature = [torch.cat(expand_feature[_ * patch_w:_ * patch_w + patch_w], dim=3) for _ in range(patch_w)]
        expand_feature = torch.cat(expand_feature, dim=2)

        # --> B input_channel*6 im_h im_w
        im_merge = torch.cat([expand_feature, im1, im2, im1-im2, im2-im1, torch.abs(im1-im2)], dim=1)
        im_merge = self.MixFFN(im_merge)

        # --> B input_channel*6 im_h im_w
        mask = self.ffn_and_project(im_merge)

        # --> B 1 im_h im_w --> B im_h im_w
        return mask.squeeze()
