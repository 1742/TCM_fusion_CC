import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import os
import sys
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
If you want to use these function to draw figure, your effect files instruction must like this:
effect----
       |____ method1
         |____effect_0.json
         |____effect_1.json
         |____effect_2.json
         |____effect_3.json
         |____effect_4.json
       |____ method2
         |____ ...
"""


def predict_entropy(mu: torch.Tensor):
    """

    :param mu:
        预测了n_sample次的预测值，维度为(bs, n_sample, n_cls)
    :return:
    """
    with torch.no_grad():
        p = torch.mean(torch.softmax(mu, dim=2), dim=1)

        return torch.sum((-1 * p * torch.log(p)), dim=1)


def ROC_and_AUC(predict: torch.Tensor, label: torch.Tensor, refer_labels: list = None, softmax: bool = True,
                smooth: bool = True, num_points: int = 50):
    """
    计算该数据集的TRP、FRP、AUC

    :param predict:
        未经softmax或sigmoid的模型输出
    :param label:
        实际标签，序号编码，内部会自动转为独热
    :param refer_labels:
        参考标签
    :param smooth:
        是否平滑ROC曲线，使用线性插值平滑
    :param num_points:
        选择平滑ROC后生效，插值点数量

    :return:
        各类别各阈值下的TRP、FRP，以及各类别AUC
    """

    n_cls = len(refer_labels)
    # 因为label_binarize不会将二分类标签做成二维向量，之后使用roc_curve时会报错。。。
    if n_cls == 2:
        n_cls = 1

    # 是否对预测进行softmax
    if softmax:
        predict = torch.argmax(torch.softmax(predict, dim=1), dim=1).cpu()
    else:
        predict = torch.argmax(predict, dim=1).cpu()
    if n_cls == 1:
        predict = np.expand_dims(predict, axis=-1)
    # 根据输入的标签类型转化成序号标签
    if label.max() == 1 and len(refer_labels) != 2:
        label = torch.argmax(label, dim=1).long().cpu()
    else:
        label = label.long().cpu()

    if refer_labels:
        label = label_binarize(label, classes=refer_labels)
    else:
        label = label_binarize(label, classes=torch.max(label))

    # 分别计算各类别的ROC曲线，若为二分类，只有一条曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], predict[:, i])
        if smooth:
            tpr[i] = np.interp(np.linspace(0, 1, num_points), fpr[i], tpr[i])
            tpr[i][0] = 0.0
            fpr[i] = np.linspace(0, 1, num_points)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def Visualization(evaluation, save_path: str = None, save_name: str = None):
    index = list(evaluation.keys())
    index.remove('epoch')
    if save_name is None:
        save_name = 'train_result.png'
    if 'fpr' in index:
        index.remove('fpr')
        index.remove('tpr')
        index.remove('AUC')

    col = len(index) // 2 + 1

    epoch = range(1, evaluation['epoch']+1)
    for i, k in enumerate(index):
        plt.subplot(2, col, i+1)
        plt.plot(epoch, evaluation[k][0], label='train', color='b')
        plt.plot(epoch, evaluation[k][1], label='val', color='r')
        plt.title('train' + k)
        plt.xlabel('epoch')
        plt.ylabel(k)
        plt.legend()
        plt.grid(1)

    if save_path:
        plt.savefig(os.path.join(save_path, save_name))

    plt.show()


def plot_LossCurve(effect_path: str, save_path: str = None, save_name: str = None):
    if 'effect.json' in effect_path:
        with open(effect_path, 'r', encoding='utf-8') as f:
            effect = json.load(f)

        epochs = effect['epoch']
        loss = effect['loss'][0]

        plt.figure()
        plt.plot(epochs, loss)
    else:
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        loss = []

        labels = os.listdir(effect_path)
        if 'gradcam.png' in labels:
            labels.remove('gradcam.png')
        effects_path = [os.path.join(effect_path, e + '\\train\\effect_4.json') for e in labels]

        for effect_path in effects_path:
            try:
                with open(effect_path, 'r', encoding='utf-8') as f:
                    effect = json.load(f)

                    loss.append(effect['loss'][1])
            except FileNotFoundError:
                continue

        plt.figure()
        for i, l in enumerate(loss):
            plt.plot(range(len(l)), l, color=colors[i], label=labels[i])
        plt.legend()

    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(1)

    if save_path:
        plt.savefig(os.path.join(save_path, save_name))

    plt.show()


def plot_ROC(fpr: [dict, list], tpr: [dict, list], AUC: [dict, list], model_name: list = None):
    model_num = len(model_name)

    # 创建画布
    plt.figure()
    # 设定线宽
    lw = 2
    if model_num != 1:
        for k in range(model_num):
            if model_name:
                name = model_name[k]
            else:
                name = k
            plt.plot(fpr[k], tpr[k], lw=lw, label='ROC curve model: {}  (AUC = {:.2f})'.format(name, AUC[k]))
    else:
        plt.plot(fpr, tpr, lw=lw, label='AUC = {:.2f}'.format(AUC))
    # 参考线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # 坐标大小
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 横纵坐标和标题，标签
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def draw_multi_model_ROC(runs_files: str, is_k: bool = True):
    model_names = os.listdir(runs_files)
    if 'gradcam.png' in model_names:
        model_names.remove('gradcam.png')
    if 'ROC.png' in model_names:
        model_names.remove('ROC.png')
    model_names.remove('resnet34_triplet')
    model_names.remove('se_resnet34_triplet')

    if is_k:
        fpr = []
        tpr = []
        AUC = []
        for model_name in model_names:
            test_effect_path = os.path.join(runs_files, model_name + '\\train\\effect_4.json')
            with open(test_effect_path, 'r', encoding='utf-8') as f:
                effect = json.load(f)
            index = np.argmax(np.array(effect['acc'][1]))
            fpr.append(effect['fpr'][index])
            tpr.append(effect['tpr'][index])
            AUC.append(effect['AUC'][index])
    else:
        fpr = []
        tpr = []
        AUC = []
        for model_name in model_names:
            test_effect_path = os.path.join(runs_files, model_name + '\\test\\effect.json')
            with open(test_effect_path, 'r', encoding='utf-8') as f:
                effect = json.load(f)
            fpr.append(effect['fpr'])
            tpr.append(effect['tpr'])
            AUC.append(effect['AUC'])

    plot_ROC(fpr, tpr, AUC, model_names)


def plot_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, refer_label: dict, model_name: str, xlim=None,
                    ylim=None, save_path: str = None):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    label_name = list(refer_label.keys())

    plt.figure(figsize=(10, 10))
    for i in range(len(refer_label)):
        inds = np.where(labels == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i], label=label_name[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.title('{} t-SNE'.format(model_name))

    if save_path:
        plt.savefig(save_path + '\\t-SNE.png')
        print('Successfully save result in {}'.format(save_path))

    plt.show()


def get_embeddings(dataloader: DataLoader, batch_size: int, model: torch.nn.Module):
    # if batch_size < 30:
    #     print('batch_size must larger than 30!')
    #     sys.exit(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    embeddings = []
    labels = []

    model.eval()
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('loading')
        with torch.no_grad():
            for i, (_, face_img, tongue_img, label) in enumerate(dataloader):
                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels.append(label)

                embedding = model.get_embeddings(tongue_img)
                embeddings.append(embedding)

                pbar.update(1)

    embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
    labels = torch.cat(labels, dim=0).cpu().detach().numpy()

    # 设置固定随机种子，保证每次生成的散点图的稳定性
    np.random.seed(0)
    tsne = TSNE(random_state=0, perplexity=10)
    embeddings = tsne.fit_transform(embeddings)
    # 归一化各特征值
    # embeddings = (embeddings - np.min(embeddings, axis=0, keepdims=True)) / (np.max(embeddings, axis=0, keepdims=True) - np.min(embeddings, axis=0, keepdims=True))

    return embeddings, labels


if __name__ == '__main__':
    # 绘制单个模型训练曲线
    # effect_path = r'your effect path\effect_0.json'
    #
    # with open(effect_path, 'r', encoding='utf-8') as f:
    #     effect = json.load(f)
    #
    # Visualization(effect)

    # 绘制ROC
    runs_path = r'your effect path'
    draw_multi_model_ROC(runs_path)

    # runs_path = r'your effect path'
    # plot_LossCurve(runs_path)

