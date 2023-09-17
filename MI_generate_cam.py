import sys

import torch
from torch import nn
import torch.nn.functional as F
from model.SE_Resnet.SE_Resnet import se_resnet34
from model.fusion.fusion import FusionNet
from tools.load_MI_model import MI_model

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle, label_encoder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tools.gradcam.gradcam import GradCAM, GradCAMpp
from tools.gradcam.utils import visualize_cam, denormalize

import os
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


data_path = r'your data path'
data_path_txt = r'../data/img_names.txt'
save_path = r'your save path'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The predict will run in {} ...'.format(device))


def load_weights(model: nn.Module, pretrained_path: str, device: torch.device):
    # 加载权重
    if os.path.exists(pretrained_path):
        # 加载模型权重文件
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint)
        print('Successfully load pretrained model from {}'.format(pretrained_path))
    else:
        print('model parameters files is not exist!')
        sys.exit(0)
    model.to(device)


def generate_gradcam_MI(
        device: torch.device,
        refer_labels: dict,
        models: dict,
        gradcam_dict: dict,
        test_data: tuple,
        save_path: str = None,
        mean: list = None,
        std: list = None,
        beta: float = 0.5
):
    img_name, face_img, tongue_img, label = test_data
    face_img = face_img.to(device, dtype=torch.float).unsqueeze(0)
    tongue_img = tongue_img.to(device, dtype=torch.float).unsqueeze(0)

    # 计算各模型预测结果
    pred = dict()
    with torch.no_grad():
        for model_name, model in models.items():
            model.eval()
            pred[model_name] = refer_labels[model(face_img, tongue_img)['pred'].item()]

    result = {'face': [], 'tongue': []}

    if mean is not None and std is not None:
        face_img_denormalize = denormalize(face_img, mean, std)
        tongue_img_denormalize = denormalize(tongue_img, mean, std)
    else:
        face_img_denormalize = face_img
        tongue_img_denormalize = tongue_img

    for gradcam in gradcam_dict.values():
        face_gradcam, tongue_gradcam = gradcam.values()
        face_mask, _ = face_gradcam(face_img_denormalize)
        tongue_mask, _ = tongue_gradcam(tongue_img_denormalize)
        # 获取cam和效果图
        face_heatmap, face_cam_result = visualize_cam(face_mask, face_img_denormalize, beta=beta)
        tongue_heatmap, tongue_cam_result = visualize_cam(tongue_mask, tongue_img_denormalize, beta=beta)
        result['face'].append(torch.stack([face_img_denormalize.squeeze().cpu(), face_heatmap, face_cam_result], 0))
        result['tongue'].append(torch.stack([tongue_img_denormalize.squeeze().cpu(), tongue_heatmap, tongue_cam_result], 0))
    # 创建网格图，方便对比
    face_result = make_grid(torch.cat(result['face'], 0), nrow=3)
    tongue_result = make_grid(torch.cat(result['tongue'], 0), nrow=3)

    if save_path:
        save_image(face_result, save_path + '\\face_gradcam.png')
        save_image(tongue_result, save_path + '\\tongue_gradcam.png')

    print('Successfully save grad-cam picture in {}'.format(save_path))
    print('img_name:', img_name)
    print('pred:', pred)
    print('label:', refer_labels[label])


if __name__ == '__main__':
    labels = os.listdir(data_path)
    if 'img_names.txt' in labels:
        labels.remove('img_names.txt')

    # 用于在生成图片时打上标签
    refer_labels = dict()
    labels = label_encoder(labels)
    items = labels.items()
    for value, key in items:
        refer_labels[key] = value

    # 划分数据集
    with open(data_path_txt, 'r', encoding='utf-8') as f:
        img_info = f.readlines()
    print("Successfully read img names from {}".format(data_path))

    # 打乱数据集
    # img_info = shuffle(img_info, 2)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformers = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_datasets = MyDatasets(data_path, labels, img_info, transformers)
    index = random.randint(0, len(img_info))
    print('index:', index)
    # 325 354 3
    test_data = test_datasets.__getitem__(index)

    # 原图尺寸及高清回复倍率
    ori_img_size = (224, 224)
    print('origin image size(included restore):', ori_img_size)

    model_name = ['SE_Resnet34_baseline', 'FusionNet']
    print('model_name:', model_name)

    pretrained_path = [
        r'./model/SE_Resnet/mi_se_resnet34.pth',
        r'./model/fusion/fusionnet.pth'
    ]

    models = dict()
    # 创建gradcam_dict
    gradcam_dict = dict()

    model_0 = MI_model('se_resnet34', num_classes=2)
    load_weights(model_0, pretrained_path[0], device)
    model_dict_0 = {
        'face': dict(model_type='ResNet', arch=model_0.face_net, layer_name='layer4', input_size=ori_img_size),
        'tongue': dict(model_type='ResNet', arch=model_0.tongue_net, layer_name='layer4', input_size=ori_img_size)
    }
    models[model_name[0]] = model_0
    gradcam_dict[model_name[0]] = {'face': GradCAM(model_dict_0['face']), 'tongue': GradCAM(model_dict_0['tongue'])}

    model_1 = FusionNet('se_resnet34', 2, dropout=0.3)
    load_weights(model_1, pretrained_path[3], device)
    model_dict_1 = {
        'face': dict(model_type='ResNet', arch=model_1.backbone[0], layer_name='layer4', input_size=ori_img_size),
        'tongue': dict(model_type='ResNet', arch=model_1.backbone[1], layer_name='layer4', input_size=ori_img_size)
    }
    models[model_name[1]] = model_1
    gradcam_dict[model_name[1]] = {'face': GradCAM(model_dict_1['face']), 'tongue': GradCAM(model_dict_1['tongue'])}

    print('gradcam_dict:', gradcam_dict.keys())

    beta = 0.3
    print('beta:', beta)

    generate_gradcam_MI(
        device=device,
        refer_labels=refer_labels,
        models=models,
        gradcam_dict=gradcam_dict,
        test_data=test_data,
        save_path=save_path,
        mean=mean,
        std=std,
        beta=beta
    )
