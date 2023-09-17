import torch
from torch import nn
from torch.functional import F

from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

import os
from tqdm import tqdm
import sys
import math
from tools.dataloader import DataPrefetcher


def load_weights(device: torch.device, model: nn.Module, pretrained_path: str):
    # 加载权重
    if pretrained_path:
        if os.path.exists(pretrained_path):
            loaded_weights = []
            not_exist_weights = []
            # 加载模型权重文件
            checkpoint = torch.load(pretrained_path, map_location=device)
            # 遍历模型层次结构
            for name, param in model.named_parameters():
                if name in checkpoint and 'fc' not in name:
                    # 逐层加载权重
                    # param.data.copy_(checkpoint[name])
                    loaded_weights.append(name)
                else:
                    try:
                        del checkpoint[name]
                    except KeyError:
                        pass
                    not_exist_weights.append(name)
            print('loaded_weights:\n', loaded_weights)
            print('not_exist_weights:\n', not_exist_weights)
            if len(loaded_weights) == 0:
                print('Please check your weights file. The model loaded nothing!')
                sys.exit(0)
            model.load_state_dict(checkpoint, strict=False)
            print('Successfully load pretrained model from {}'.format(pretrained_path))
        else:
            print('model parameters files is not exist!')
            sys.exit(0)


def enable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def fit_normal(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(tongue_img)

                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(output, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    # face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(tongue_img)

                    loss = criterion(output, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(output, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_combined_uncertainty(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.model, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                mu, sigma = model(tongue_img, uncertainty='combined')

                loss = criterion(mu, sigma, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0.
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    # face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    mu, sigma = model(tongue_img, uncertainty='combined')

                    loss = criterion(mu, sigma, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(mu, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(fpr.tolist())
        val_tpr.append(tpr.tolist())
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_tripletmodel(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算样本数量
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_triplets_num = []
    val_loss = []
    val_triplets_num = []
    # early_stop判断指标
    lowest_val_loss = 1000.
    last_val_loss = 0.
    flag = 0

    # 加载权重
    load_weights(device, model.model, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0.
        per_triplets_num = 0
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                embeddings = model(tongue_img, uncertainty='dropout')

                loss, triplets_num = criterion(embeddings, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录每批次预测和标签
                per_loss += loss.item()
                per_triplets_num += triplets_num

                pbar.update(1)

        per_loss /= train_step
        print('[train] loss: {:.2f} triplets_num: {}'.format(per_loss, per_triplets_num))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_triplets_num.append(per_triplets_num)

        # 验证
        model.eval()

        per_loss = 0.
        per_triplets_num = 0
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    # face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    embeddings = model(tongue_img, uncertainty='dropout')

                    loss, triplets_num = criterion(embeddings, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    per_triplets_num += triplets_num

                    pbar.update(1)

        per_loss /= val_step
        print('[validation] loss: {:.2f} triplets_num: {}'.format(per_loss, per_triplets_num))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_triplets_num.append(per_triplets_num)

        if per_loss < lowest_val_loss:
            lowest_val_loss = per_loss
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if per_loss > last_val_loss:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_loss = per_loss

    print('The lastest val_loss:', per_loss)
    print('The recorded lowest val_acc:', lowest_val_loss)
    choice = input('Would you wanted to save the lastest weights?[y/n]')
    if choice == 'y':
        torch.save(model.state_dict(), save_path)
        print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'triplets_num': [train_triplets_num, val_triplets_num]
    }


def fit_triplet_and_CE_model(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_triplets_num = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    val_triplets_num = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 0.
    flag = 0

    # 加载权重
    load_weights(device, model.model, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                mu, embeddings = model(tongue_img, uncertainty='dropout')

                loss, triplets_num = criterion(mu, embeddings, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录每批次预测和标签
                per_loss += loss.item()
                per_triplets_num += triplets_num
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, per_triplets_num))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        train_triplets_num.append(per_triplets_num)

        # 验证
        model.eval()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    # face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    mu, embeddings = model(tongue_img, uncertainty='dropout')

                    loss, triplets_num = criterion(mu, embeddings, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    per_triplets_num += triplets_num
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(mu, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC, per_triplets_num))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_triplets_num.append(per_triplets_num)
        val_fpr.append(fpr.tolist())
        val_tpr.append(tpr.tolist())
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc,
        'triplets_num':[train_triplets_num, val_triplets_num]
    }


def fit_triplet_and_unc_model(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_triplets_num = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    val_triplets_num = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 0.
    flag = 0

    # 加载权重
    load_weights(device, model.model, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                mu, sigma, embeddings = model(tongue_img, uncertainty='combined')

                loss, triplets_num = criterion(mu, sigma, embeddings, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录每批次预测和标签
                per_loss += loss.item()
                per_triplets_num += triplets_num
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, per_triplets_num))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        train_triplets_num.append(per_triplets_num)

        # 验证
        model.eval()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    # face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    mu, sigma, embeddings = model(tongue_img, uncertainty='combined')

                    loss, triplets_num = criterion(mu, sigma, embeddings, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    per_triplets_num += triplets_num
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(mu, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC, per_triplets_num))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_triplets_num.append(per_triplets_num)
        val_fpr.append(fpr.tolist())
        val_tpr.append(tpr.tolist())
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc,
        'triplets_num':[train_triplets_num, val_triplets_num]
    }


def fit_MI_Model_normal(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.face_net, pretrained_path)
    load_weights(device, model.tongue_net, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img)

                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(output, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img)

                    loss = criterion(output, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(output, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_DSNModel(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.face_net, pretrained_path)
    load_weights(device, model.tongue_net, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img)

                loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(output['prob'], dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img)

                    loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = output['prob'][:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(output['prob'], dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_MI_Unc_Model(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.model.face_net, pretrained_path)
    load_weights(device, model.model.tongue_net, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0.
        pred = np.zeros(train_num)
        pred_entropy = 0.
        pred_sigma = 0.
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img, uncertainty='combined')

                loss = criterion(output['logits'], output['aleatoric'], labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                pred[index_range[0]:index_range[1]] = torch.argmax(output['mu'], dim=1).detach().cpu().detach().numpy()
                pred_entropy += output['epistemic'].cpu().detach().mean().item()
                pred_sigma += output['aleatoric'].cpu().detach().mean().item()
                true[index_range[0]:index_range[1]] = labels.detach().cpu().detach().numpy()

                pbar.update(1)

        per_loss /= train_step
        pred_entropy /= train_step
        pred_sigma /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} epistemic: {:.2f} aleatoric: {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, pred_entropy, pred_sigma))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()
        enable_dropout(model.model.face_net)
        enable_dropout(model.model.tongue_net)

        per_loss = 0.
        softmax = np.zeros(val_num)
        pred = np.zeros(val_num)
        pred_entropy = 0.
        pred_sigma = 0.
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img, uncertainty='combined')

                    loss = criterion(output['logits'], output['aleatoric'], labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    softmax[index_range[0]:index_range[1]] = output['mu'][:, 1].cpu().detach().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(output['mu'], dim=1).detach().cpu().detach().numpy()
                    pred_entropy += output['epistemic'].cpu().detach().mean().item()
                    pred_sigma += output['aleatoric'].cpu().detach().mean().item()
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().detach().numpy()

                    pbar.update(1)

        per_loss /= val_step
        pred_entropy /= val_step
        pred_sigma /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC：{:.2f} epistemic: {:.2f} aleatoric: {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC, pred_entropy, pred_sigma))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(fpr.tolist())
        val_tpr.append(tpr.tolist())
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_MI_triplet_and_CE_model(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_triplets_num = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    val_triplets_num = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 0.
    flag = 0

    # 加载权重
    load_weights(device, model.face_net, pretrained_path)
    load_weights(device, model.tongue_net, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                mu, embeddings = model(face_img, tongue_img, uncertainty='dropout')

                loss, triplets_num = criterion(mu, embeddings, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录每批次预测和标签
                per_loss += loss.item()
                per_triplets_num += triplets_num
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, per_triplets_num))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        train_triplets_num.append(per_triplets_num)

        # 验证
        model.eval()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    mu, embeddings = model(face_img, tongue_img, uncertainty='dropout')

                    loss, triplets_num = criterion(mu, embeddings, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    per_triplets_num += triplets_num
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(mu, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC, per_triplets_num))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_triplets_num.append(per_triplets_num)
        val_fpr.append(fpr.tolist())
        val_tpr.append(tpr.tolist())
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc,
        'triplets_num':[train_triplets_num, val_triplets_num]
    }


def fit_MI_triplet_and_unc_model(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True
):

    # 计算类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_triplets_num = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    val_triplets_num = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 0.
    flag = 0

    # 加载权重
    load_weights(device, model.face_net, pretrained_path)
    load_weights(device, model.tongue_net, pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                mu, sigma, embeddings = model(face_img, tongue_img, uncertainty='combined')

                loss, triplets_num = criterion(mu, sigma, embeddings, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录每批次预测和标签
                per_loss += loss.item()
                per_triplets_num += triplets_num
                true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, per_triplets_num))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        train_triplets_num.append(per_triplets_num)

        # 验证
        model.eval()

        per_loss = 0.
        per_triplets_num = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    mu, sigma, embeddings = model(face_img, tongue_img, uncertainty='combined')

                    loss, triplets_num = criterion(mu, sigma, embeddings, labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    per_triplets_num += triplets_num
                    true[index_range[0]:index_range[1]] = labels.detach().cpu().numpy()
                    softmax[index_range[0]:index_range[1]] = torch.softmax(mu, dim=1)[:, 1].detach().cpu().numpy()
                    pred[index_range[0]:index_range[1]] = torch.argmax(mu, dim=1).detach().cpu().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC： {:.2f} triplets_num: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC, per_triplets_num))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_triplets_num.append(per_triplets_num)
        val_fpr.append(fpr.tolist())
        val_tpr.append(tpr.tolist())
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc,
        'triplets_num':[train_triplets_num, val_triplets_num]
    }


def fit_FusionNet_without_adaptive_weights(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.backbone[0], pretrained_path)
    load_weights(device, model.backbone[1], pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img)

                loss = criterion(output['concat_logits'], labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img)

                    loss = criterion(output['concat_logits'], labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                    softmax[index_range[0]:index_range[1]] = output['prob'][:, 1].cpu().detach().numpy()
                    pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC: {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_FusionNet_without_DSN(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.backbone[0], pretrained_path)
    load_weights(device, model.backbone[1], pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img)

                loss = criterion(output['concat_logits'], labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img)

                    loss = criterion(output['concat_logits'], labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                    softmax[index_range[0]:index_range[1]] = output['prob'][:, 1].cpu().detach().numpy()
                    pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC: {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_FusionNet_without_DSN_adaptive_weights(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.backbone[0], pretrained_path)
    load_weights(device, model.backbone[1], pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img)

                loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img)

                    loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels)

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                    softmax[index_range[0]:index_range[1]] = output['prob'][:, 1].cpu().detach().numpy()
                    pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC: {:.2f}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc
    }


def fit_FusionNet(
        device: torch.device,
        model: nn.Module,
        train_num: int,
        val_num: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        optimizer,
        criterion,
        save_path: str = None,
        lr_schedule=None,
        pretrained_path: str = None,
        early_stop: bool = True,
):
    # 计算样本数量和类别数
    n_cls = len(refer_labels)

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_dsn_weights = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_fpr = []
    val_tpr = []
    val_auc = []
    val_dsn_weights = []
    # early_stop判断指标
    best_val_acc = 0.
    last_val_acc = 100.
    flag = 0

    # 加载权重
    load_weights(device, model.backbone[0], pretrained_path)
    load_weights(device, model.backbone[1], pretrained_path)
    model.to(device)

    train_step = len(train_dataloader)
    val_step = len(val_dataloader)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1)

    for epoch in range(epochs):
        # 训练
        model.train()

        per_loss = 0
        pred = np.zeros(train_num)
        true = np.zeros(train_num)
        per_dsn_weights = np.zeros((train_num, 3))
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (_, face_img, tongue_img, labels) in enumerate(train_dataloader):
                index_range = [i * batch_size, i * batch_size + labels.size(0)]

                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                output = model(face_img, tongue_img)

                loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels, output['dsn_weights'])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录每批次预测和标签
                per_loss += loss.item()
                true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()
                per_dsn_weights[index_range[0]:index_range[1], :] = output['dsn_weights'].cpu().detach().numpy()

                pbar.update(1)

        per_loss /= train_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        per_dsn_weights = np.mean(per_dsn_weights, axis=0)
        print('[train] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} dsn_weights: {}'.format(per_loss,
            acc * 100, precision * 100, recall * 100, f1 * 100, per_dsn_weights))

        # 更新学习率
        lr_schedule.step()

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print('Warning! The grad disappeared! Pause the train!')
            epoch -= 1
            break

        # 记录每训练次平均指标
        train_loss.append(per_loss)
        train_acc.append(acc)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        train_dsn_weights.append(per_dsn_weights.tolist())

        # 验证
        model.eval()

        per_loss = 0
        pred = np.zeros(val_num)
        softmax = np.zeros(val_num)
        true = np.zeros(val_num)
        per_dsn_weights = np.zeros((val_num, 3))
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (_, face_img, tongue_img, labels) in enumerate(val_dataloader):
                    index_range = [i * batch_size, i * batch_size + labels.size(0)]

                    face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(face_img, tongue_img)

                    loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels, output['dsn_weights'])

                    # 记录每批次预测和标签
                    per_loss += loss.item()
                    true[index_range[0]:index_range[1]] = labels.cpu().detach().numpy()
                    softmax[index_range[0]:index_range[1]] = output['prob'][:, 1].cpu().detach().numpy()
                    pred[index_range[0]:index_range[1]] = output['pred'].cpu().detach().numpy()
                    per_dsn_weights[index_range[0]:index_range[1], :] = output['dsn_weights'].cpu().detach().numpy()

                    pbar.update(1)

        per_loss /= val_step
        acc = accuracy_score(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        f1 = f1_score(true, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(true, softmax)
        AUC = roc_auc_score(true, softmax)
        per_dsn_weights = np.mean(per_dsn_weights, axis=0)
        print('[validation] loss: {:.2f} acc: {:.2f} precision: {:.2f} recall: {:.2f} f1: {:.2f} AUC: {:.2f} dsn_weights: {}'.format(
            per_loss, acc * 100, precision * 100, recall * 100, f1 * 100, AUC, per_dsn_weights))

        # 记录每训练次平均指标
        val_loss.append(per_loss)
        val_acc.append(acc)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        val_fpr.append(list(fpr))
        val_tpr.append(list(tpr))
        val_auc.append(AUC)
        val_dsn_weights.append(per_dsn_weights.tolist())

        if acc > best_val_acc:
            best_val_acc = acc
            # 仅保存验证集上效果最好的权重
            if save_path:
                torch.save(model.state_dict(), save_path)
                print('Successfully saved the best model weights in {}'.format(save_path))

        if early_stop:
            # 计数
            if acc < last_val_acc:
                flag += 1
            # 若连续10次验证集准确率没有提升则停止训练
            if flag == 10:
                print('The train had no improved last 10 epochs. Stop train.')
                break
            # 更新上次准确率
            last_val_acc = acc

    # print('The lastest val_acc:', acc)
    # print('The recorded best val_acc:', best_val_acc)
    # choice = input('Would you wanted to save the lastest weights?[y/n]')
    # if choice == 'y':
    #     torch.save(model.state_dict(), save_path)
    #     print('Successfully saved the lastest model weights in {}'.format(save_path))

    print('Train End.')

    return {
        'epoch': epoch + 1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1], 'fpr': val_fpr, 'tpr': val_tpr, 'AUC': val_auc,
        'dsn_weights': [train_dsn_weights, val_dsn_weights]
    }

