import json
import os

import numpy as np


def calculate_result(effects_path: str):
    if 'effect.json' in effects_path:
        with open(effects_path, 'r', encoding='utf-8') as f:
            effect = json.load(f)

        train_loss = effect['loss'][0]
        train_acc = effect['acc'][0]
        train_precision = effect['precision'][0]
        train_recall = effect['recall'][0]
        train_f1 = effect['f1'][0]

        val_loss = effect['loss'][1]
        val_acc = effect['acc'][1]
        val_precision = effect['precision'][1]
        val_recall = effect['recall'][1]
        val_f1 = effect['f1'][1]

        print('训练损失:', sum(train_loss) / len(train_loss))
        print('训练准确率:', sum(train_acc) / len(train_acc))
        print('训练精确率:', sum(train_precision) / len(train_precision))
        print('训练召回率:', sum(train_recall) / len(train_recall))
        print('训练f1率:', sum(train_f1) / len(train_f1))
        print('\n')
        max_index = np.argmax(val_acc)
        print('最小验证损失:', val_loss[max_index])
        print('最佳验证准确率:', val_acc[max_index])
        print('最佳验证精确率:', val_precision[max_index])
        print('最佳验证召回率:', val_recall[max_index])
        print('最佳验证f1率:', val_f1[max_index])
    else:
        test_loss = []
        test_acc = []
        test_precision = []
        test_recall = []
        test_f1 = []

        effects_path = [os.path.join(effects_path, e) for e in os.listdir(effects_path)]
        for effect_path in effects_path:
            with open(effect_path, 'r', encoding='utf-8') as f:
                effect = json.load(f)

            max_index = np.argmax(effect['acc'][1])
            test_loss.append(effect['loss'][1][max_index])
            test_acc.append(effect['acc'][1][max_index])
            test_precision.append(effect['precision'][1][max_index])
            test_recall.append(effect['recall'][1][max_index])
            test_f1.append(effect['f1'][1][max_index])

        test_loss_mean = np.mean(test_loss)
        test_loss_std = np.std(test_loss)

        test_acc_mean = np.mean(test_acc)
        test_acc_std = np.std(test_acc)

        test_precision_mean = np.mean(test_precision)
        test_precision_std = np.std(test_precision)

        test_recall_mean = np.mean(test_recall)
        test_recall_std = np.std(test_recall)

        test_f1_mean = np.mean(test_f1)
        test_f1_std = np.std(test_f1)

        print('k折测试损失:{} +- {}'.format(test_loss_mean, test_loss_std))
        print('k折测试准确率:{} +- {}'.format(test_acc_mean, test_acc_std))
        print('k折测试精确率:{} +- {}'.format(test_precision_mean, test_precision_std))
        print('k折测试召回率:{} +- {}'.format(test_recall_mean, test_recall_std))
        print('k折测试f1率:{} +- {}'.format(test_f1_mean, test_f1_std))


if __name__ == '__main__':
    effect_path = r'../runs/MI/fusionnet_without_adaptive_weights/train'

    calculate_result(effect_path)

