"""
卫星图像建筑物变化识别
data: 数据集由配对的图像构成，图像没有给出时序 图像
    ***整个数据集小于4G，可是将数据集直接加载到RAM里加速训练
train       # 2500 pairs  512*512
    .../image1  # tif format 0-255
    .../image2  # tif format
    .../label1  # png format 0-1
test
    ... # 500 pairs without label

标签前景定义：
    1）建筑物变化才算，植物变化不算
    2）有一定高度（形成阴影的才算）的建筑物才算
        # 见2.tif
        # 阴影R通道的变化远大于G和B通道
    3）建筑物的阴影不算变化
    4）部分标注的面积很小
    5）标注一般由多边形构成，曲面很少
    6）标注有明显错误的地方, 比例未统计
        # 见1009.tif
    7）标注多为块状，几乎没有散点状的，如果有细密的建筑变化情况，这将整块区域标注为一个
    8）G通道的变化与标注的关系弱
        # G通道的变化一般由植被变化引起


一种简单的方式是设计一个模型输入为图像对，输出为变化的二值图，进行端到端的训练
当前最优的方法（changer)指出，在图像编码的过程中将图像对的特征进行互换（空间和通道）是一种有效的操作

在这个问题中有两个关键的问题
1）对变化的识别：由于两张图片拍摄的条件不同，因此即使物理对象没有发生变化，图像任然是有差别的
2）识别有效的变化：当前的任务是识别建筑物的变化，需要在所有的变化中将建筑物的变化识别出来

"""
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def read_pair(im_name: str):
    root_dir = '../CD_dataset/train'
    numpy_im = []
    format_ = {'Image1': '.tif', 'Image2': '.tif', 'label1': '.png'}
    for s in ['Image1', 'Image2', 'label1']:
        numpy_im.append(cv2.imread(os.path.join(root_dir, s, im_name + format_[s])))

    return numpy_im


def check_1():
    for name in os.listdir('../CD_dataset/train/Image1'):
        pair = read_pair(name.split('.')[0])
        plt.subplot(1,4,1)
        plt.imshow(pair[0])
        plt.subplot(1,4,2)
        plt.imshow(pair[1])
        plt.subplot(1,4,3)
        plt.imshow(np.abs(pair[0]-pair[1]))
        plt.subplot(1,4,4)
        plt.imshow(pair[2]*255)
        plt.title(name.split('.')[0])
        plt.show()


if __name__ == '__main__':
    check_1()