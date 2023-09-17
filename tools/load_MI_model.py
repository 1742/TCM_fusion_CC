import sys

import torch
from torch import nn
from torch.nn import init
from model.ResNet.resnet import resnet18, resnet34, resnet50
from model.SE_Resnet.SE_Resnet import se_resnet18, se_resnet34, se_resnet50, SE_Block

import os
import json


class MI_model(nn.Module):
    def __init__(self, backbone: str = 'resnet34', num_classes: int = 1000, include_top: bool = True, out_layer: bool = False,
                 is_sigmoid: bool = False, dropout: float = 0.):
        super(MI_model, self).__init__()
        if backbone.lower() == 'resnet18':
            backbone = [
                resnet18(include_top=False),
                resnet18(include_top=False)
            ]
        elif backbone.lower() == 'resnet34':
            backbone = [
                resnet34(include_top=False),
                resnet34(include_top=False)
            ]
        elif backbone.lower() == 'resnet50':
            backbone = [
                resnet50(include_top=False),
                resnet50(include_top=False)
            ]
        elif backbone.lower() == 'se_resnet18':
            backbone = [
                se_resnet18(include_top=False),
                se_resnet18(include_top=False)
            ]
        elif backbone.lower() == 'se_resnet34':
            backbone = [
                se_resnet34(include_top=False),
                se_resnet34(include_top=False)
            ]
        elif backbone.lower() == 'se_resnet50':
            backbone = [
                se_resnet50(include_top=False),
                se_resnet50(include_top=False)
            ]

        self.face_net = backbone[0]
        self.tongue_net = backbone[1]

        self.fc_cells = self.face_net.fc_cells + self.tongue_net.fc_cells

        # 是否使用全连接层
        self.include_top = include_top
        if include_top:
            self.avepool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

            if num_classes == 2 and is_sigmoid:
                self.fc = nn.Linear(self.fc_cells, 1)
            else:
                self.fc = nn.Linear(self.fc_cells, num_classes)

            # 是否使用softmax或sigmoid
            self.out_layer = out_layer
            if out_layer:
                if num_classes == 2 and is_sigmoid:
                    self.out_layer = nn.Sigmoid()
                else:
                    self.out_layer = nn.Softmax(dim=1)

        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def forward(self, x, y):
        x = self.face_net(x)
        y = self.tongue_net(y)

        if self.include_top:
            feature = torch.cat((x, y), dim=1)
            feature = self.flatten(self.avepool(feature))

            if self.dropout:
                feature = self.dropout(feature)

            pred = self.fc(feature)

            if self.out_layer:
                pred = self.out_layer(pred)

            return pred

        return x, y

    def get_embeddings(self, x, y):
        x = self.face_net(x)
        y = self.tongue_net(y)

        feature = torch.cat((x, y), dim=1)
        feature = self.flatten(self.avepool(feature))

        return feature

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


class DSNModel(MI_model):
    def __init__(self, backbone: str = 'resnet34', n_cls: int = 1000, include_top: bool = True, out_layer: bool = True,
                 is_sigmoid: bool = False, dropout: float = 0.):
        super(DSNModel, self).__init__(backbone=backbone, num_classes=n_cls, include_top=include_top,
                                       out_layer=out_layer, is_sigmoid=is_sigmoid, dropout=dropout)
        if n_cls == 2 and is_sigmoid:
            self.x_out = nn.Linear(self.face_net.fc_cells, 1)
            self.y_out = nn.Linear(self.face_net.fc_cells, 1)
        else:
            self.x_out = nn.Linear(self.face_net.fc_cells, n_cls)
            self.y_out = nn.Linear(self.tongue_net.fc_cells, n_cls)

    def forward(self, x, y):
        x = self.face_net(x)
        y = self.tongue_net(y)

        if self.include_top:
            x_logits = self.x_out(self.flatten(self.avepool(x)))
            y_logits = self.y_out(self.flatten(self.avepool(y)))

            concat = torch.cat((x, y), dim=1)
            feature = self.flatten(self.avepool(concat))

            if self.dropout:
                feature = self.dropout(feature)
            concat_logits = self.fc(feature)

            output = {'x_logits': x_logits, 'y_logits': y_logits, 'concat_logits': concat_logits}

            if self.out_layer:
                prob = self.out_layer(concat_logits)
                output['prob'] = prob

            return output

        return x, y


if __name__ == '__main__':
    from tools.MyLoss import DSNLoss_cls

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DSNModel(n_cls=2).to(device)
    print(model)

    x = torch.randint(255, (8, 3, 448, 448)).float().to(device)
    y = torch.randint(255, (8, 3, 448, 448)).float().to(device)
    labels = torch.randint(2, (8,)).to(device)

    output = model(x / 255., y / 255.)
    # embeddings = model.get_embeddings(x / 255., y / 255.)
    print(output['x_logits'], output['x_logits'].size())
    print(output['y_logits'], output['y_logits'].size())
    print(output['concat_logits'], output['concat_logits'].size())
    print(output['prob'], output['prob'].size())

    criterion = DSNLoss_cls()
    loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels)
    print(loss)



