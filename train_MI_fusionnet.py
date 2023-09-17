import torch
from torch import nn
from torch.functional import F
from tools.load_MI_model import MI_model, MI_SE_Concat_Model
from model.fusion.fusion import FusionNet
from tools.MyLoss import AdaptiveDSNLoss_cls

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, BalancedBatchSampler, DataPrefetcher, k_fold_maker, label_encoder
from tools.fit import fit_FusionNet

import os
import sys
import json
import numpy as np
from tools.metrics import Visualization

"""
If you want to use the dataloader, the data files instruction must like this:
data----
     |____ sx
       |____ face
         |____ face image
       |____ tongue
         |____ tongue image
     |____ xx
       |____ face
         |____ face image
       |____ tongue
         |____ tongue image

And the data txt like this:
img_name1.png label
img_name2.png label
...
"""

data_path = r'your data path'
data_path_txt = r'./data/data.txt'
pretrained_path = r'./model/checkpoints/resnet34-b627a593.pth'
save_path = r'your save path'
effect_path = r'your save result path'

learning_rate = 1e-3
weight_decay = 1e-4
epochs = 50
num_classes = 2
n_sample = 32
batch_size = n_sample * num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    labels = os.listdir(data_path)
    if 'img_names.txt' in labels:
        labels.remove('img_names.txt')
    if 'ph' in labels:
        labels.remove('ph')

    # 在计算混淆矩阵时需传入参考标签的序号，防止输入的predict和labels不含有某一类别
    refer_labels = list(label_encoder(labels).values())

    batch_sampler = True

    model_name = 'FusionNet'
    model = FusionNet(backbone='se_resnet34', n_cls=len(refer_labels), dropout=0.3)
    # pretrained_path = None

    early_stop = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = AdaptiveDSNLoss_cls()
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # lr_schedule = None
    print('model:\n', model)
    print('model_name:', model_name)
    print('refer_labels:', refer_labels)
    print('epoch:', epochs)
    print('batch size:', batch_size)
    print('batch sampler:', batch_sampler)
    print('learning rate:', learning_rate)
    print('weight decay:', weight_decay)
    print('loss:', criterion)
    print('optimizer:', optimizer)
    print('lr_schedule:', lr_schedule)
    print('early_stop:', early_stop)
    print('The train will run in {} ...'.format(device))

    folds = k_fold_maker(data_path_txt)

    train_transformers = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transformers = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for k, dataset in folds.items():
        if k == 'train_num' or k == 'test_num':
            break

        print('K then cross-verifies: the {} rule'.format(k))

        train_data_info, val_data_info = dataset['train'], dataset['test']

        train_datasets = MyDatasets(data_path, labels, train_data_info, train_transformers)
        val_datasets = MyDatasets(data_path, labels, val_data_info, val_transformers)

        # 设置线程
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        if batch_sampler:
            # 设置采样器，确保每批次含有所有类别的样本
            train_batchsampler = BalancedBatchSampler(train_datasets.all_labels, len(refer_labels), n_sample)
            val_batchsampler = BalancedBatchSampler(val_datasets.all_labels, len(refer_labels), n_sample)
            # 使用batchsampler
            train_dataloader = DataLoader(train_datasets, batch_sampler=train_batchsampler, num_workers=nw, pin_memory=True)
            val_dataloader = DataLoader(val_datasets, batch_sampler=val_batchsampler, num_workers=nw, pin_memory=True)
            # 计算各数据集数量
            train_num = len(train_dataloader) * batch_size
            val_num = len(val_dataloader) * batch_size
        else:
            # 计算各数据集数量
            train_num = len(train_datasets)
            val_num = len(val_datasets)
            # 普通dataloader
            train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
            val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        print('train_num:', train_num)
        print('val_num:', val_num)

        effect = fit_FusionNet(
            device=device,
            model=model,
            train_num=train_num,
            val_num=val_num,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            refer_labels=refer_labels,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            save_path=save_path,
            lr_schedule=lr_schedule,
            pretrained_path=pretrained_path,
            early_stop=early_stop
        )

        if not os.path.exists(effect_path):
            os.mkdir(effect_path)
        with open(effect_path + '\\effect_{}.json'.format(k), 'w', encoding='utf-8') as f:
            f.write(json.dumps(effect))

        # Visualization(effect, save_path=effect_path, save_name='train_result_{}'.format(k))

        # 重置模型参数
        model._init_weights()
        # 重置优化器参数
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 重置lr_schedule参数
        if lr_schedule:
            lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # break


