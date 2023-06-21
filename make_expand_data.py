# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

dest_dir = 'DATA/expand_dataset'
data_path_1 = 'DATA/CD_Data_GZ'
data_path_2 = 'DATA/CD_dataset'
data_path_3 = 'DATA/LEVIR_CD'

sample_im_1 = cv2.imread(data_path_1 + '/T1/P_GZ_test1_2013_1221_Level_18.tif')
sample_im_2 = cv2.imread(data_path_2 + '/train/Image1/1.tif')
sample_im_3 = cv2.imread(data_path_3 + '/train/Image1/train_1.png')
sample_label_1 = cv2.imread(data_path_1 + '/labels_change/P_GZ_test1_2013_2018.png')
sample_label_2 = cv2.imread(data_path_2 + '/train/label1/1.png')
sample_label_3 = cv2.imread(data_path_3 + '/train/label1/train_1.png')

"""
2 中的数据直接复制
1 对大图进行旋转操作， 然后在图像中随机裁剪512大小的图若干
3 对大图进行旋转操作， 然后在图像中随机裁剪512大小的图若干
step1:设目标矩形边长为a, 先在大图中按照步长为b，边长为a*2**0.5将大图裁剪为若干小图
step2:对每一张小图在四个角上随机取两张图，再以将矩形随机旋转两次，取图像中心的矩形两张，得到4张裁剪图像
GZ:
    ori_im: 5224 4936 3 0-255
    block_size: 724
    crop_size: 512
    step: 724
    label: 0-255 
LEVIR_CD:
    ori_im 1024 1024 3 0-255
    block_size: 724
    crop_size: 512
    step: 300
"""


def crop_block(numpy_im1, numpy_im2, label, crop_size: int):
    """
    numpy_im1: H W 3
    numpy_im2: H W 3
    label: H W 3
    return [block_1, ...] h w C
    """
    crop_ims = []   # [[im1, im2, label], ...]
    H, W, C = numpy_im1.shape
    corner = [[0, 0], [W-crop_size-1, H-crop_size-1], [W-crop_size-1, 0], [0, H-crop_size-1]]
    for w, h in random.sample(corner, 2):
        crop_ims.append([
            numpy_im1[w:w + crop_size, h:h + crop_size, ::],
            numpy_im2[w:w + crop_size, h:h + crop_size, ::],
            label[w:w + crop_size, h:h + crop_size, ::]
        ])

    for i in range(2):
        angle = random.random() * 360
        rotate_center = (W / 2, H / 2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算图像新边界
        new_w = int(H * np.abs(M[0, 1]) + W * np.abs(M[0, 0]))
        new_h = int(H * np.abs(M[0, 0]) + W * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - W) / 2
        M[1, 2] += (new_h - H) / 2

        new_im1 = cv2.warpAffine(numpy_im1, M, (new_w, new_h))
        new_im2 = cv2.warpAffine(numpy_im2, M, (new_w, new_h))
        new_label = cv2.warpAffine(label, M, (new_w, new_h))
        new_im1 = new_im1[new_h//2 - crop_size // 2: new_h//2 - crop_size // 2 + crop_size, new_h//2 - crop_size // 2:new_h//2 - crop_size // 2 + crop_size,:]
        new_im2 = new_im2[new_h//2 - crop_size // 2: new_h//2 - crop_size // 2 + crop_size, new_h//2 - crop_size // 2:new_h//2 - crop_size // 2 + crop_size,:]
        new_label = new_label[new_h//2 - crop_size // 2: new_h//2 - crop_size // 2 + crop_size, new_h//2 - crop_size // 2:new_h//2 - crop_size // 2 + crop_size,:]

        crop_ims.append([
            new_im1,
            new_im2,
            new_label
        ])

    return crop_ims


def make_expand_dataset():
    pair_name_count = 1
    val_ratio = 0.1
    # GZ
    for block_path in glob.glob(r'D:\change_detection\DATA\CD_Data_GZ\T1\*.tif'):
        block1 = cv2.imread(block_path)
        block2 = cv2.imread(block_path.replace('T1', 'T2'))
        label = cv2.imread(block_path.replace('T1', 'labels_change').replace('.tif', '.png'))
        H, W, C = block1.shape
        step = 724
        for w in range(W//step):
            for h in range(H//step):
                crop_im = crop_block(
                    block1[h * step: h * step + 724, w * step: w * step + 724, ::],
                    block2[h * step: h * step + 724, w * step: w * step + 724, ::],
                    label[h * step: h * step + 724, w * step: w * step + 724, ::],
                    crop_size=512
                )
                for im1, im2, label1 in crop_im:
                    if random.random() > val_ratio:
                        cv2.imwrite(os.path.join(dest_dir, 'train', 'Image1', '{}.tif'.format(pair_name_count)), im1)
                        cv2.imwrite(os.path.join(dest_dir, 'train', 'Image2', '{}.tif'.format(pair_name_count)), im2)
                        cv2.imwrite(os.path.join(dest_dir, 'train', 'label1', '{}.png'.format(pair_name_count)), label1/255)
                        pair_name_count += 1
                    else:
                        cv2.imwrite(os.path.join(dest_dir, 'val', 'Image1', '{}.tif'.format(pair_name_count)), im1)
                        cv2.imwrite(os.path.join(dest_dir, 'val', 'Image2', '{}.tif'.format(pair_name_count)), im2)
                        cv2.imwrite(os.path.join(dest_dir, 'val', 'label1', '{}.png'.format(pair_name_count)), label1/255)
                        pair_name_count += 1

    # LEVIR_CD
    for block_path in glob.glob(r'D:\change_detection\DATA\LEVIR_CD\*\Image1\*.png'):
        block1 = cv2.imread(block_path)
        block2 = cv2.imread(block_path.replace('Image1', 'Image2'))
        label = cv2.imread(block_path.replace('Image1', 'label1'))
        step = 300
        for w in range(2):
            for h in range(2):
                crop_im = crop_block(
                    block1[h * step: h * step + 724, w * step: w * step + 724, ::],
                    block2[h * step: h * step + 724, w * step: w * step + 724, ::],
                    label[h * step: h * step + 724, w * step: w * step + 724, ::],
                    crop_size=512
                )
                for im1, im2, label1 in crop_im:
                    if random.random() > val_ratio:
                        cv2.imwrite(os.path.join(dest_dir, 'train', 'Image1', '{}.tif'.format(pair_name_count)), im1)
                        cv2.imwrite(os.path.join(dest_dir, 'train', 'Image2', '{}.tif'.format(pair_name_count)), im2)
                        cv2.imwrite(os.path.join(dest_dir, 'train', 'label1', '{}.png'.format(pair_name_count)),
                                    label1 / 255)
                        pair_name_count += 1
                    else:
                        cv2.imwrite(os.path.join(dest_dir, 'val', 'Image1', '{}.tif'.format(pair_name_count)), im1)
                        cv2.imwrite(os.path.join(dest_dir, 'val', 'Image2', '{}.tif'.format(pair_name_count)), im2)
                        cv2.imwrite(os.path.join(dest_dir, 'val', 'label1', '{}.png'.format(pair_name_count)),
                                    label1 / 255)
                        pair_name_count += 1

    # CD_dataset
    for im_path in glob.glob(r'D:\change_detection\DATA\CD_dataset\train\Image1\*.tif'):
        im1 = cv2.imread(im_path)
        im2 = cv2.imread(im_path.replace('Image1', 'Image2'))
        label1 = cv2.imread(im_path.replace('Image1', 'label1').replace('.tif', '.png'))
        label1 = label1 * 255
        if random.random() > val_ratio:
            cv2.imwrite(os.path.join(dest_dir, 'train', 'Image1', '{}.tif'.format(pair_name_count)), im1)
            cv2.imwrite(os.path.join(dest_dir, 'train', 'Image2', '{}.tif'.format(pair_name_count)), im2)
            cv2.imwrite(os.path.join(dest_dir, 'train', 'label1', '{}.png'.format(pair_name_count)),
                        label1 / 255)
            pair_name_count += 1
        else:
            cv2.imwrite(os.path.join(dest_dir, 'val', 'Image1', '{}.tif'.format(pair_name_count)), im1)
            cv2.imwrite(os.path.join(dest_dir, 'val', 'Image2', '{}.tif'.format(pair_name_count)), im2)
            cv2.imwrite(os.path.join(dest_dir, 'val', 'label1', '{}.png'.format(pair_name_count)),
                        label1 / 255)
            pair_name_count += 1

    for im_path in glob.glob(r'D:\change_detection\DATA\CD_dataset\val\Image1\*.tif'):
        im1 = cv2.imread(im_path)
        im2 = cv2.imread(im_path.replace('Image1', 'Image2'))
        label1 = cv2.imread(im_path.replace('Image1', 'label1').replace('.tif', '.png'))
        label1 = label1 * 255
        if random.random() > val_ratio:
            cv2.imwrite(os.path.join(dest_dir, 'train', 'Image1', '{}.tif'.format(pair_name_count)), im1)
            cv2.imwrite(os.path.join(dest_dir, 'train', 'Image2', '{}.tif'.format(pair_name_count)), im2)
            cv2.imwrite(os.path.join(dest_dir, 'train', 'label1', '{}.png'.format(pair_name_count)),
                        label1 / 255)
            pair_name_count += 1
        else:
            cv2.imwrite(os.path.join(dest_dir, 'val', 'Image1', '{}.tif'.format(pair_name_count)), im1)
            cv2.imwrite(os.path.join(dest_dir, 'val', 'Image2', '{}.tif'.format(pair_name_count)), im2)
            cv2.imwrite(os.path.join(dest_dir, 'val', 'label1', '{}.png'.format(pair_name_count)),
                        label1 / 255)
            pair_name_count += 1

    print('total', pair_name_count)


def show_sample(pair):
    plt.subplot(1,3,1)
    plt.imshow(pair[0])
    plt.subplot(1,3,2)
    plt.imshow(pair[1])
    plt.subplot(1,3,3)
    plt.imshow(pair[2])
    plt.show()


if __name__ == '__main__':
    make_expand_dataset()
    pass
