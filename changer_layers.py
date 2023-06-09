# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


class SpatialExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(W) % self.p == 0
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class FDAF(nn.Module):
    def __init__(self,
                 in_channels, ):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=True, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1, x2):
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1

        output = torch.cat([x1_feat, x2_feat])
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)

        return output


class MixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        layers = [fc1, pe_conv, self.activate, fc2]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


if __name__ == '__main__':
    net = ChannelExchange()
    x1 = torch.rand(2,3,512,512)
    x2 = torch.rand(2,3,512,512)
    x1, x2 = net(x1, x2)
    print(x1.shape)
    print(x2.shape)