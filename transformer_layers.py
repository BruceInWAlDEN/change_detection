# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


class Mlp(nn.Module):
    """
    B patch_num dim -> B patch_num dim
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class Attention(nn.Module):
    """
    B patch_num dim -> B patch_num dim
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5   # for different len of head_dim, Q@K could be huge which is bad for softmax.
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3 B num_heads patch_num C//num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # B num_heads patch_num C//num_heads
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   # B num_heads patch_num C//num_heads  @  B num_heads C//num_heads patch_num
        attn = attn.softmax(dim=-1)     # B num_heads patch_num patch_num(softmax)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # B num_heads patch_num patch_num(softmax) @ B num_heads patch_num C//num_heads
        # -> B num_heads patch_num C//num_heads
        # -> B num_heads C//num_heads patch_num
        # -> B patch_num C
        x = self.proj(x)

        return x


class Block(nn.Module):
    """
    B patch_num dim -> B patch_num dim
    """
    def __init__(self, dim, num_head, mlp_ratio):
        super().__init__()
        self.multi_head_attention = Attention(dim, num_head)
        self.ffn = Mlp(dim, dim*mlp_ratio)
        self.norm = nn.LayerNorm(eps=1e-6, normalized_shape=dim)

    def forward(self, x):
        x = x + self.multi_head_attention(x)    # B patch_num dim
        x = self.norm(x)
        x = x + self.ffn(x)  # B patch_num dim
        x = self.norm(x)

        return x