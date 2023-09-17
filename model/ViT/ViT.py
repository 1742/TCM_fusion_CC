import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, patch_dim: int = 512, nhead: int = 8, dim_head: int = 64, dropout: float = 0.):
        super(MultiHeadAttention, self).__init__()
        inner = nhead * dim_head
        self.head = nhead
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(patch_dim, 3 * inner, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(inner, patch_dim)
        self.rearrange = Rearrange('b h n d -> b n (h d)')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        atten = self.softmax(dots)
        atten = torch.einsum('b h i j, b h j d -> b h i d', atten, v)

        out = self.rearrange(atten)
        out = self.dropout(self.out(out))

        return out


class CrossAttention(nn.Module):
    def __init__(self, patch_dim: int = 512, nhead: int = 8, dim_head: int = 64, dropout: float = 0.):
        super(CrossAttention, self).__init__()
        inner = nhead * dim_head
        self.head = nhead
        self.scale = dim_head ** -0.5

        self.query = nn.Linear(patch_dim, inner, bias=False)
        self.key = nn.Linear(patch_dim, inner, bias=False)
        self.value = nn.Linear(patch_dim, inner, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(inner, patch_dim)
        self.rearrange = Rearrange('b h n d -> b n (h d)')
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head), [q, k, v])

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        atten = self.softmax(dots)
        atten = torch.einsum('b h i j, b h j d -> b h i d', atten, v)

        out = self.rearrange(atten)
        out = self.dropout(self.out(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, dropout: float = 0.):
        super(FeedForward, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feedforward(x)


class Transformer(nn.Module):
    def __init__(self, patch_dim: int = 512, nhead: int = 8, dim_head: int = 64, dim_feedforward: int = 2048,
                 dropout: float = 0., norm_first: bool = False, is_cross_atten: bool = False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.is_cross_atten = is_cross_atten

        self.atten = MultiHeadAttention(patch_dim, nhead, dim_head, dropout)
        self.layer_norm1 = nn.LayerNorm(patch_dim)

        if is_cross_atten:
            self.cross_atten = CrossAttention(patch_dim, nhead, dim_head, dropout)
            self.layer_norm_ = nn.LayerNorm(patch_dim)

        self.feedforward = FeedForward(patch_dim, dim_feedforward, dropout)
        self.layer_norm2 = nn.LayerNorm(patch_dim)

    def forward(self, q, k=None, v=None):
        if self.is_cross_atten:
            assert k is not None or v is not None, 'You did not pass key or value in.'
            if self.norm_first:
                q = self.atten(self.layer_norm1(q)) + q
                hidden_out = self.cross_atten(*self.layer_norm_(torch.cat([q, k, v], dim=1)).chunk(3, dim=1)) + q
                hidden_out = self.feedforward(self.layer_norm2(hidden_out)) + hidden_out
            else:
                q = self.layer_norm1(self.atten(q)) + q
                hidden_out = self.layer_norm_(self.cross_atten(q, k, v)) + q
                hidden_out = self.layer_norm2(self.feedforward(hidden_out)) + hidden_out
        else:
            if self.norm_first:
                x = self.atten(self.layer_norm1(q)) + q
                hidden_out = self.feedforward(self.layer_norm2(x)) + x
            else:
                x = self.layer_norm1(self.atten(q)) + q
                hidden_out = self.layer_norm2(self.feedforward(x)) + x

        return hidden_out


class ViT(nn.Module):
    def __init__(self, input_size: [int, list, tuple] = [3, 224, 224], patch_size: [list, tuple] = [16, 16],
                 n_cls: int = 1000, dim_feedforward: int = 2048, n_head: int = 8, dim_head: int = 64, n_layer: int = 6,
                 dropout: float = 0.1, norm_first: bool = False):
        super(ViT, self).__init__()
        self.input_size = self._get_input_size(input_size)
        self.n_cls = n_cls

        img_channel, img_weight, img_height = self.input_size
        patch_weight, patch_height = self._get_patch_size(patch_size)

        assert img_weight % patch_weight == 0 or img_height % patch_height == 0, 'Expect img size can be divisible by patch size'

        self.n_patch = (img_weight // patch_weight) * (img_height // patch_height)
        self.patch_dim = img_channel * patch_weight * patch_height

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(img_channel, self.patch_dim, (patch_weight, patch_height), (patch_weight, patch_height)),
            Rearrange('b c h w -> b (h w) c')
        )

        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patch, self.patch_dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer_encoder = self._make_encoder(self.patch_dim, n_head, dim_head, n_layer, dim_feedforward, dropout, norm_first)

        self.classifier = nn.Linear(self.patch_dim, n_cls)
        self.out_layer = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)

        logits = self.classifier(x)
        prob = self.out_layer(logits)
        pred = torch.argmax(prob, dim=1, keepdim=True)

        return {'logits': logits, 'prob': prob, 'pred': pred}

    def get_feature(self, x):
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        x = self.dropout(x)
        feature = self.transformer_encoder(x)

        return feature

    def _get_input_size(self, size):
        if isinstance(size, int):
            return size, size, 1
        else:
            if len(size) == 2:
                return size[0], size[1], 1
            else:
                return size

    def _get_patch_size(self, size):
        assert len(size) == 2, 'patch_size dimension wrong! Excpect list or tuple of (patch_weight, patch_height, patch_dim)'
        return size

    def _make_encoder(self, patch_dim, n_head, dim_head, n_layer, dim_feedforward, dropout, norm_first):
        layers = []
        for _ in range(n_layer):
            layers.append(Transformer(patch_dim, n_head, dim_head, dim_feedforward, dropout, norm_first))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = torch.randn((8, 3, 224, 224)).to(device)
    labels = torch.randint(1000, (8,)).to(device)

    model = ViT(n_head=12, n_layer=12)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    output = model(img)
    print(output['logits'], output['logits'].size())
    print(output['prob'], output['prob'].size())
    print(output['pred'], output['pred'].size())

    # loss = criterion(output['logits'], labels)
    #
    # loss.backward()


#
    # loss.backward()


