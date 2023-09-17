import sys

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler

from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import KFold


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels, n_classes, n_samples):
        super(BalancedBatchSampler, self).__init__(batch_size=n_classes*n_samples, sampler=RandomSampler, drop_last=False)
        """
        :param labels:
            所有样本的标签
        :param n_classes:
            类别数
        :param n_samples:
            采样次数，输出的batch_size为n_classes * n_samples
        """
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        # 每个类别的样本序号
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # 打乱每个类别中的样本顺序
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # 记录最后使用的各类别的样本序号
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                # 从各个类别的序号中抽出n_sample个样本
                # self.used_label_indices_count[class_]记录该类别最后使用的样本序号
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                # 若该类别已被使用完，则打乱该类的序号重置该类最后使用的样本序号为0
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            # 打乱顺序
            np.random.shuffle(indices)
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class MyDatasets(Dataset):
    def __init__(self, data_path: str, label: list, img_info: [list, np.ndarray], transformers=None):
        super(MyDatasets, self).__init__()
        self.data_path = data_path
        # 包含所有样本的文件名、标签
        # ['img_name label\n', ...]
        self.img_info = img_info
        # 序号编码
        self.labels = label_encoder(label)
        # 独热编码
        # self.labels = one_hot_encoder(label)
        # 所有样本的标签
        self.all_labels = torch.Tensor([self.labels[i.strip().split(' ')[1]] for i in img_info])
        # 数据增强
        self.transformers = transformers

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        # 除去换行符并以空格分割出图片名字和标签
        img_name, label = self.img_info[index].strip().split(' ')

        face_img_path = os.path.join(os.path.join(self.data_path, label), os.path.join('face', img_name))
        tongue_img_path = os.path.join(os.path.join(self.data_path, label), os.path.join('tongue', img_name))
        try:
            face_img = Image.open(face_img_path).convert('RGB')
        except FileNotFoundError:
            wrong_suffix = img_name.split('.')[-1]
            if wrong_suffix == 'jpg':
                suffix = 'png'
            else:
                suffix = 'jpg'
            face_img = Image.open(face_img_path.replace(wrong_suffix, suffix)).convert('RGB')
        try:
            tongue_img = Image.open(tongue_img_path).convert('RGB')
        except FileNotFoundError:
            wrong_suffix = img_name.split('.')[-1]
            if wrong_suffix == 'jpg':
                suffix = 'png'
            else:
                suffix = 'jpg'
            tongue_img = Image.open(tongue_img_path.replace(wrong_suffix, suffix)).convert('RGB')

        # 数据增强
        if self.transformers:
            face_img = self.transformers(face_img)
            tongue_img = self.transformers(tongue_img)
        # 返回序号编码
        label = self.labels[label]
        # 返回独热编码
        # label = self.labels[label]

        return img_name, face_img, tongue_img, label


class DataPrefetcher():
    """用于在cuda上加载数据，加速数据加载 emmmm重置的时候更耗时间......."""
    def __init__(self, loader):
        # 加载器长度
        self.length = len(loader)
        # 记录原始加载器，用于重置迭代器
        self.ori_loader = loader
        # 将加载器转化成迭代器
        self.loader = iter(loader)
        # 创建显卡工作流
        self.stream = torch.cuda.Stream()
        # 预加载下一批数据
        self.preload()

    def __len__(self):
        return self.length

    def preload(self):
        try:
            self.img_name, self.face_img, self.tongue_img, self.label = next(self.loader)
        except StopIteration:
            self.img_name = None
            self.face_img = None
            self.tongue_img = None
            self.label = None
            return

        with torch.cuda.stream(self.stream):
            # self.img_name = self.img_name.cuda(non_blocking=True)
            self.face_img = self.face_img.cuda(non_blocking=True)
            self.tongue_img = self.tongue_img.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)
            self.face_img = self.face_img.float()
            self.tongue_img = self.tongue_img.float()
            self.label = self.label.long()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img_name = self.img_name
        face_img = self.face_img
        tongue_img = self.tongue_img
        label = self.label
        if img_name is not None:
            face_img.record_stream(torch.cuda.current_stream())
            tongue_img.record_stream(torch.cuda.current_stream())
            label.record_stream(torch.cuda.current_stream())
            self.preload()
            return img_name, face_img, tongue_img, label
        else:
            self._reset()
            raise StopIteration

    def __iter__(self):
        return self

    def _reset(self):
        # 重置迭代器
        self.loader = iter(self.ori_loader)
        self.preload()


def k_fold_maker(data_path_txt: str, n_splits: int = 5, shuffle: bool = False):
    # 实例化K则分割器
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)

    # 读取数据txt
    with open(data_path_txt, 'r', encoding='utf-8') as f:
        img_info = f.readlines()
    img_info = np.array(img_info)

    folds = dict()
    for i, (train_incides, test_incides) in enumerate(kfold.split(img_info)):
        folds[i] = {'train': img_info[train_incides], 'test': img_info[test_incides]}
    folds['train_num'] = len(train_incides)
    folds['test_num'] = len(test_incides)

    return folds


# 独热编码
def one_hot_encoder(label: [list, torch.Tensor]):
    labels = {}
    cls_num = len(label)
    for i, cls in enumerate(label):
        k = torch.zeros(cls_num)
        k[i] = 1
        labels[cls] = k

    return labels


def label_encoder(label: list):
    # 数字编码
    labels = {}
    for i, cls in enumerate(label):
        labels[cls] = i

    return labels


def shuffle(data: list, times: int = 2):
    for _ in range(times):
        np.random.shuffle(data)
    return data


if __name__ == '__main__':
    data_path = r'G:\zzl\constitution_data\data_v2'
    data_path_txt = r'G:\zzl\Constitution_Classification\data\img_names_v2.txt'

    refer_labels = label_encoder(['sx', 'xx'])
    n_cls = len(refer_labels)
    n_sample = 16
    batch_size = n_cls * n_sample

    label = os.listdir(data_path)
    if 'img_names.txt' in label:
        label.remove('img_names.txt')

    if not os.path.exists(data_path_txt):
        with open(data_path_txt, 'w', encoding='utf-8') as f:
            for cls in label:
                cls_path = os.path.join(data_path, cls)
                for img in os.listdir(os.path.join(cls_path, 'tongue')):
                    f.write(img + ' ' + cls)
                    f.write('\n')
        print('Successfully generated img names file in {}!'.format(data_path_txt))

    # with open(img_names_path, 'r', encoding='utf-8') as f:
    #     img_info = f.readlines()
    # print("Successfully read img names in {}".format(data_path))

    # 保存打乱顺序后的数据
    # img_info = shuffle(img_info, 4)
    # with open(img_names_path, 'w', encoding='utf-8') as f:
    #     for img in img_info:
    #         f.write(img)
    # print('Successfully shuffle img names file in {}'.format(img_names_path))

    transformers = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 15)),
        transforms.ToTensor()
    ])

    folds = k_fold_maker(data_path_txt)

    test_datasets = MyDatasets(data_path, label, folds[0]['train'], transformers)
    batchsampler = BalancedBatchSampler(test_datasets.all_labels, n_classes=n_cls, n_samples=n_sample)
    test_dataloader = DataLoader(test_datasets, batch_sampler=batchsampler, pin_memory=True)

    _, (img_name, face_img, tongue_img, labels) = next(enumerate(test_dataloader))
    print(labels)

    face_img = face_img.cpu().detach()
    tongue_img = tongue_img.cpu().detach()

    img_name, face_img, tongue_img, label = img_name[0], face_img[0], tongue_img[0], label[0]

    plt.subplot(1, 2, 1)
    plt.imshow(face_img.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(tongue_img.permute(1, 2, 0))
    plt.show()
    print(img_name)
    print(label)



