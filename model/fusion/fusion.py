import torch
from torch import nn
from model.ResNet.resnet import resnet18, resnet34, resnet50
from model.SE_Resnet.SE_Resnet import se_resnet18, se_resnet34, se_resnet50
from model.ViT.ViT import Transformer, ViT
from einops.layers.torch import Rearrange


class FusionNet(nn.Module):
    def __init__(self, backbone: str = 'resnet34', n_cls: int = 1000, n_transformer_head: int = 8,
                 n_transformer_dim_head: int = 64, dim_trainsformer_feedforward: int = 2048,
                 n_transformer_layer: int = 1, input_size: [int, list, tuple] = [3, 224, 224],
                 patch_size: [list, tuple] = [16, 16], dim_feedforward: int = 2048, n_head: int = 8, dim_head: int = 64,
                 n_layer: int = 6, norm_first: bool = True, r: int = 4, dropout: float = 0.):
        super(FusionNet, self).__init__()
        self.n_cls = n_cls
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if backbone.lower() == 'resnet18':
            self.backbone = nn.ModuleList([
                resnet18(include_top=False),
                resnet18(include_top=False)
            ])
        elif backbone.lower() == 'resnet34':
            self.backbone = nn.ModuleList([
                resnet34(include_top=False),
                resnet34(include_top=False)
            ])
        elif backbone.lower() == 'resnet50':
            self.backbone = nn.ModuleList([
                resnet50(include_top=False),
                resnet50(include_top=False)
            ])
        if backbone.lower() == 'se_resnet18':
            self.backbone = nn.ModuleList([
                se_resnet18(include_top=False),
                se_resnet18(include_top=False)
            ])
        elif backbone.lower() == 'se_resnet34':
            self.backbone = nn.ModuleList([
                se_resnet34(include_top=False),
                se_resnet34(include_top=False)
            ])
        elif backbone.lower() == 'se_resnet50':
            self.backbone = nn.ModuleList([
                se_resnet50(include_top=False),
                se_resnet50(include_top=False)
            ])
        elif backbone.lower() == 'vit':
            self.backbone = nn.ModuleList([
                ViT(input_size, patch_size, n_cls, dim_feedforward, n_head, dim_head, n_layer, dropout, norm_first),
                ViT(input_size, patch_size, n_cls, dim_feedforward, n_head, dim_head, n_layer, dropout, norm_first)
            ])

        if 'resnet' in backbone.lower():
            self.n_patch = 7 * 7
            self.patch_dim = self.backbone[0].fc_cells
            self.rearrange = Rearrange('b c w h -> b (w h) c')
            self.position_embedding = nn.Parameter(torch.randn(1, self.n_patch, self.patch_dim))
        elif 'vit' in backbone.lower():
            self.n_patch = self.backbone[0].n_patch
            self.patch_dim = self.backbone[0].patch_dim
            self.rearrange = nn.Identity()
            self.position_embedding = torch.zeros(1, self.n_patch, self.patch_dim)
        # CrossAttention
        self.face_to_tongue = Transformer(
            self.patch_dim, n_head, dim_head, dim_feedforward, dropout, norm_first, True
        )
        self.tongue_to_face = Transformer(
            self.patch_dim, n_head, dim_head, dim_feedforward, dropout, norm_first, True
        )
        # SelfAttention
        self.transformer_layer_x = self._make_encoder(
            self.patch_dim, n_transformer_head, n_transformer_dim_head, n_transformer_layer,
            dim_trainsformer_feedforward, dropout, norm_first
        )
        self.transformer_layer_y = self._make_encoder(
            self.patch_dim, n_transformer_head, n_transformer_dim_head, n_transformer_layer,
            dim_trainsformer_feedforward, dropout, norm_first
        )
        # Adaptive Weight Concat Module
        self.adaptive_weights = nn.Sequential(
            nn.Linear(2 * self.patch_dim, 2 * self.patch_dim // r),
            nn.Linear(2 * self.patch_dim // r, 2 * self.patch_dim),
            nn.Sigmoid()
        )
        # DSN Module
        self.x_out = nn.Linear(self.patch_dim, n_cls)
        self.y_out = nn.Linear(self.patch_dim, n_cls)
        self.concat_out = nn.Linear(2 * self.patch_dim, n_cls)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x = self.backbone[0].get_feature(x)
        y = self.backbone[1].get_feature(y)
        # 调整骨干输出进行后续Transformer部分
        x = self.rearrange(x) + self.position_embedding
        y = self.rearrange(y) + self.position_embedding
        # CrossAttention
        x_ = self.tongue_to_face(y, x, x) + x
        y_ = self.face_to_tongue(x, y, y) + y
        # 使用共享transformer层分析
        x_ = self.transformer_layer_x(x_)
        y_ = self.transformer_layer_y(y_)
        # 平均池化转为一维张量
        x_ = torch.mean(x_, dim=1)
        y_ = torch.mean(y_, dim=1)
        # 计算两种模态的贡献程度
        x_w, y_w = self.adaptive_weights(torch.cat([x_, y_], dim=1)).chunk(2, dim=1)
        # 面象特征、舌象特征及自适应融合特征
        concat_ = torch.cat([x_w * x_, y_w * y_], dim=1)
        # DSN logits
        x_logits = self.x_out(x_)
        y_logits = self.y_out(y_)
        concat_logits = self.concat_out(concat_)
        # 概率及预测
        prob = self.softmax(concat_logits)
        pred = torch.argmax(prob, dim=1)

        return {
            'x_logits': x_logits, 'y_logits': y_logits, 'concat_logits': concat_logits, 'prob': prob, 'pred': pred
        }

    def get_fusion_embeddings(self, x, y):
        x = self.backbone[0].get_feature(x)
        y = self.backbone[1].get_feature(y)
        # 调整骨干输出进行后续Transformer部分
        x = self.rearrange(x) + self.position_embedding
        y = self.rearrange(y) + self.position_embedding
        # CrossAttention
        x_ = self.tongue_to_face(y, x, x) + x
        y_ = self.face_to_tongue(x, y, y) + y
        # 使用共享transformer层分析
        x_ = self.transformer_layer_x(x_)
        y_ = self.transformer_layer_y(y_)
        # 平均池化转为一维张量
        x_ = torch.mean(x_, dim=1)
        y_ = torch.mean(y_, dim=1)
        # 计算两种模态的贡献程度
        x_w, y_w = self.adaptive_weights(torch.cat([x_, y_], dim=1)).chunk(2, dim=1)
        # 面象特征、舌象特征及自适应融合特征
        concat_ = torch.cat([x_w * x_, y_w * y_], dim=1)

        return concat_

    def _make_encoder(self, patch_dim, n_head, dim_head, n_layer, dim_feedforward, dropout, norm_first):
        layers = []
        for _ in range(n_layer):
            layers.append(Transformer(patch_dim, n_head, dim_head, dim_feedforward, dropout, norm_first))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class FusionNetWithoutAdaptiveWeights(FusionNet):
    def __init__(self, backbone: str = 'resnet34', n_cls: int = 1000, n_transformer_head: int = 8,
                 n_transformer_dim_head: int = 64, dim_trainsformer_feedforward: int = 2048,
                 n_transformer_layer: int = 1, input_size: [int, list, tuple] = [3, 224, 224],
                 patch_size: [list, tuple] = [16, 16], dim_feedforward: int = 2048, n_head: int = 8, dim_head: int = 64,
                 n_layer: int = 6, norm_first: bool = True, dropout: float = 0.):
        super(FusionNetWithoutAdaptiveWeights, self).__init__(
            backbone=backbone, n_cls=n_cls, n_transformer_head=n_transformer_head,
            n_transformer_dim_head=n_transformer_dim_head, dim_trainsformer_feedforward=dim_trainsformer_feedforward,
            n_transformer_layer=n_transformer_layer, input_size=input_size, patch_size=patch_size,
            dim_feedforward=dim_feedforward, n_head=n_head, dim_head=dim_head, n_layer=n_layer, norm_first=norm_first,
            dropout=dropout
        )

        delattr(self, 'adaptive_weights')
        delattr(self, 'adaptive_dsn_weights')

    def forward(self, x, y):
        x = self.backbone[0].get_feature(x)
        y = self.backbone[1].get_feature(y)
        # 调整骨干输出进行后续Transformer部分
        x = self.rearrange(x) + self.position_embedding
        y = self.rearrange(y) + self.position_embedding
        # CrossAttention
        x_ = self.tongue_to_face(y, x, x) + x
        y_ = self.face_to_tongue(x, y, y) + y
        # 使用共享transformer层分析
        x_ = self.transformer_layer_x(x_)
        y_ = self.transformer_layer_y(y_)
        # 平均池化转为一维张量
        x_ = torch.mean(x_, dim=1)
        y_ = torch.mean(y_, dim=1)
        # 面象特征、舌象特征融合
        concat_ = torch.cat([x_, y_], dim=1)
        # DSN logits
        x_logits = self.x_out(x_)
        y_logits = self.y_out(y_)
        concat_logits = self.concat_out(concat_)
        # 概率及预测
        prob = self.softmax(concat_logits)
        pred = torch.argmax(prob, dim=1)

        return {'x_logits': x_logits, 'y_logits': y_logits, 'concat_logits': concat_logits, 'prob': prob, 'pred': pred}

    def get_fusion_embeddings(self, x, y):
        x = self.backbone[0].get_feature(x)
        y = self.backbone[1].get_feature(y)
        # 调整骨干输出进行后续Transformer部分
        x = self.rearrange(x) + self.position_embedding
        y = self.rearrange(y) + self.position_embedding
        # CrossAttention
        x_ = self.tongue_to_face(y, x, x) + x
        y_ = self.face_to_tongue(x, y, y) + y
        # 使用共享transformer层分析
        x_ = self.transformer_layer_x(x_)
        y_ = self.transformer_layer_y(y_)
        # 平均池化转为一维张量
        x_ = torch.mean(x_, dim=1)
        y_ = torch.mean(y_, dim=1)
        # 模态融合
        concat_ = torch.cat([x_, y_], dim=1)

        return concat_


if __name__ == '__main__':
    from tools.MyLoss import DSNLoss_cls

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    face = torch.randn(8, 3, 224, 224).to(device)
    tongue = torch.randn(8, 3, 224, 224).to(device)
    labels = torch.randint(2, (8,)).to(device)

    model = FusionNet(n_cls=2)
    model.to(device)
    print(model)

    output = model(face, tongue)

    print(output['concat_logits'], output['concat_logits'].size())
    print(output['prob'], output['prob'].size())

    criterion = DSNLoss_cls()
    loss = criterion([output['x_logits'], output['y_logits'], output['concat_logits']], labels)
    print(loss)

