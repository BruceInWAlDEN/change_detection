# -*- coding: utf-8 -*-
import numpy as np
from math import cos
from math import pi
import os
import torch
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from model.model import MixChanger, MixChanger_base
from tqdm import tqdm
from dataset import Mydata
import torch.nn as nn

CD_v2 = {
    'model_name': 'MixChanger_v1',
    'cuda_id': 0,
    'batch_size': 4,
    'epoch_start': 1,
    'epoch_end': 400,
    'logdir_path': '',
    'check_epoch': [_ for _ in range(400) if _ % 4 == 1],
    'recover_epoch': -1,
    'data_root': 'DATA/CD_dataset'
}


def launch(cfg):

    # set random seed
    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def fix_conv():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set seed
    set_seed(123)
    fix_conv()

    main_worker(cfg)


def main_worker(cfg):
    # print info
    print('===============================================================================================')
    print('to train encoder')
    for k, v in cfg.items():
        print(k, v)

    # metrics
    writer = SummaryWriter(os.path.join(cfg['logdir_path'], 'tf'))

    # device
    device = torch.device('cuda', index=cfg['cuda_id'])

    # model
    model = MixChanger(**MixChanger_base)
    model.to(device)

    # data
    train_data = Mydata(data_root_dir='DATA/CD_dataset', c='train')
    val_data = Mydata(data_root_dir='DATA/CD_dataset', c='test')
    train_data.batch_size = cfg['batch_size']
    val_data.batch_size = cfg['batch_size']
    train_loader = train_data.get_loader()
    val_loader = val_data.get_loader()

    # lr schedule
    sche = lr_schedule(cfg['batch_size'], max_epoch=cfg['epoch_end'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=0.05, betas=(0.9, 0.95))
    schedule = LambdaLR(optimizer, lr_lambda=sche, last_epoch=-1)

    # loss
    criterion = DiceLoss()

    # recovery
    if cfg['recover_epoch'] != -1:
        print('RECOVER FROM:: {} epoch'.format(cfg['recover_epoch']))
        pth = torch.load(os.path.join(cfg['logdir_path'], '{}_{}.pth'.format(cfg['model_name'], cfg['recover_epoch'])),
                         map_location='cpu')
        model.load_state_dict(pth['model_weights'])  # before wrap param
        optimizer.load_state_dict(pth['optimizer_weights'])
        schedule.load_state_dict(pth['schedule_weights'])
        cfg['epoch_start'] = cfg['recover_epoch'] + 1

    for epoch in range(cfg['epoch_start'], cfg['epoch_end'] + 1):

        # train
        model.train()
        loss_epoch = 0
        for im1, im2, label, sam_feature, name in tqdm(train_loader, desc='Train Epoch {}: '.format(epoch)):
            im1 = im1.to(device)
            im2 = im2.to(device)
            label = label.to(device)
            sam_feature = sam_feature.to(device)
            outputs = model(im1, im2, sam_feature)
            loss = criterion(outputs, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

            print(loss.item())

        schedule.step()

        writer.add_scalar(tag='train_epoch_loss', global_step=epoch, scalar_value=loss_epoch)
        print('train_epoch_loss', loss_epoch)

        # eval
        model.eval()
        val_loss = 0
        for im1, im2, label, sam_feature, name in tqdm(val_loader, desc='Train Epoch {}: '.format(epoch)):
            im1 = im1.to(device)
            im2 = im2.to(device)
            label = label.to(device)
            sam_feature = sam_feature.to(device)

            with torch.no_grad():
                outputs = model(im1, im2, sam_feature)
                loss = criterion(outputs, label)
            val_loss += loss.item()

        writer.add_scalar(tag='val_epoch_loss', global_step=epoch, scalar_value=val_loss)
        print('val_epoch_loss', val_loss)

        # save check ================================================================================================
        if epoch in cfg['check_epoch']:
            epoch_info = {
                'model_weights': model.state_dict(),
                'optimizer_weights': optimizer.state_dict(),
                'schedule_weights': schedule.state_dict()
            }
            torch.save(epoch_info, os.path.join(cfg['logdir_path'], '{}_{}.pth'.format(cfg['model_name'], epoch)))
            print(
                'save check: {}'.format(os.path.join(cfg['logdir_path'], '{}_{}.pth'.format(cfg['model_name'], epoch))))


def lr_schedule(batch_size, max_epoch):
    def lr_(epoch):
        # optimizer init lr should be 1
        epoch = epoch + 1
        warmup_epoch = int(max_epoch * 0.05)
        peak_lr = 1.5e-4*batch_size/256
        lr_min = 1e-7
        lr = 0
        if epoch <= warmup_epoch:
            lr = epoch * peak_lr/warmup_epoch
        if max_epoch >= epoch > warmup_epoch:
            lr = peak_lr * (1 + cos(pi * (epoch-warmup_epoch) / (max_epoch-warmup_epoch))) / 2
        if lr <= lr_min:
            lr = lr_min

        return lr

    return lr_


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


if __name__ == '__main__':
    launch(CD_v2)
