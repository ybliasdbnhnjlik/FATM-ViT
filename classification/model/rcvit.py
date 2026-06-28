"""
Code for CAS-ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np
from einops import rearrange, repeat
import itertools
import os
import copy

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

# ======================================================================================================================
def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


def _expand_stage_config(value, num_stages):
    if isinstance(value, (list, tuple)):
        assert len(value) == num_stages
        return list(value)
    return [value] * num_stages

class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class LiteSE(nn.Module):
    def __init__(self, dim, rd_ratio=0.125):
        super().__init__()
        hidden = max(8, int(dim * rd_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            norm_layer(dim),
            act_layer(),
        )

    def forward(self, x):
        return self.network(x)
    

class FrequencyExtrator(nn.Module):
    def __init__(self, out_dim=1, hidden_dim=16):
        super().__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2 * out_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        xm = x.mean(dim=1, keepdim=True)
        gx = xm[:, :, :, 1:] - xm[:, :, :, :-1]
        gy = xm[:, :, 1:, :] - xm[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        lap = -4.0 * xm
        lap = lap + F.pad(xm[:, :, 1:, :], (0, 0, 0, 1))
        lap = lap + F.pad(xm[:, :, :-1, :], (0, 0, 1, 0))
        lap = lap + F.pad(xm[:, :, :, 1:], (0, 1, 0, 0))
        lap = lap + F.pad(xm[:, :, :, :-1], (1, 0, 0, 0))

        mean = self.avg(xm)
        mean_sq = self.avg(xm ** 2)
        var = mean_sq - mean ** 2

        freq = torch.cat([grad, lap, var], dim=1)
        gate_beta = self.proj(freq)
        gate, beta = gate_beta.chunk(2, dim=1)
        return gate, beta


class LargeKernelPerception(nn.Module):
    def __init__(self, dim, kernel_size=7, channelwise=True):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2

        if channelwise:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), stride=1, padding=(pad, 0), groups=dim, bias=False),
                nn.BatchNorm2d(dim),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            )

    def forward(self, x):
        return self.block(x)
    

class SmallKermelFocus(nn.Module):
    def __init__(self, dim, kernel_size=3, channelwise=True):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2

        if channelwise:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(dim),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            )

    def forward(self, x):
        return self.block(x)
    

class FLSSpatialOperation(nn.Module):
    def __init__(self, dim, kernel_size=7, freq_hidden_dim=16, channelwise=True):
        super().__init__()
        self.channelwise = channelwise
        out_dim = dim if channelwise else 1
        self.large = LargeKernelPerception(dim, kernel_size=kernel_size, channelwise=channelwise)
        self.small = SmallKermelFocus(dim, kernel_size=3, channelwise=channelwise)
        self.freq = FrequencyExtrator(out_dim=out_dim, hidden_dim=freq_hidden_dim)

        if channelwise:
            self.out_proj = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(dim),
            )
        else:
            self.out_proj = nn.Identity()

    def forward(self, x):
        large = self.large(x)
        small = self.small(x)
        gate, beta = self.freq(x)

        if self.channelwise:
            beta = torch.sigmoid(beta)
            gate = torch.sigmoid(gate)
            mix = large - beta * small
            mix = gate * mix + (1.0 - gate) * x
            mix = self.out_proj(mix)
            return mix

        attn = torch.sigmoid(large + gate - torch.tanh(beta) * small)
        return x * attn


class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0., use_fls=False, fls_kernel_size=7,
                 freq_hidden_dim=16, fls_channelwise=True):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        
        if use_fls:
            spatial = FLSSpatialOperation(
                dim,
                kernel_size=fls_kernel_size,
                freq_hidden_dim=freq_hidden_dim,
                channelwise=fls_channelwise,
            )
        else:
            spatial = SpatialOperation(dim)
        
        self.oper = nn.Sequential(
            spatial,
            ChannelOperation(dim),
        )

        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        base = q + k
        gate = torch.tanh(self.oper(base))
        out = v + self.dwc(gate * v)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class AdditiveBlock(nn.Module):
    """
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d,
                 use_fls=False, fls_kernel_size=7, freq_hidden_dim=16, use_local=True,
                 fls_channelwise=True,
                 layer_scale_init_value=1e-4):
        super().__init__()
        if use_local:
            self.local_perception = LocalIntegration(dim, act_layer=act_layer, norm_layer=norm_layer)
        else:
            self.local_perception = nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(
            dim,
            attn_bias=attn_bias,
            proj_drop=drop,
            use_fls=use_fls,
            fls_kernel_size=fls_kernel_size,
            freq_hidden_dim=freq_hidden_dim,
            fls_channelwise=fls_channelwise,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.local_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.local_scale.unsqueeze(-1).unsqueeze(-1) * self.local_perception(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

def Stage(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, attn_bias=False, drop=0., drop_path_rate=0.,
          use_fls_stages=None, fls_kernel_size=7, freq_hidden_dim=16, use_local=True,
          fls_channelwise=True, layer_scale_init_value=1e-4):
    """
    """
    blocks = []

    use_fls = False if use_fls_stages is None else bool(use_fls_stages[index])
    stage_kernel_size = 7 if fls_kernel_size is None else fls_kernel_size[index]
    stage_mlp_ratio = mlp_ratio[index]
    stage_freq_hidden_dim = freq_hidden_dim[index]
    stage_use_local = bool(use_local[index])
    stage_fls_channelwise = bool(fls_channelwise[index])
    stage_layer_scale = layer_scale_init_value[index]

    total_blocks = sum(layers)

    for block_idx in range(layers[index]):
        if total_blocks > 1:
            block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        else:
            block_dpr = 0.

        blocks.append(
            AdditiveBlock(
                dim, mlp_ratio=stage_mlp_ratio, attn_bias=attn_bias, drop=drop, drop_path=block_dpr,
                act_layer=act_layer, norm_layer=nn.BatchNorm2d,
                use_fls=use_fls, fls_kernel_size=stage_kernel_size,
                freq_hidden_dim=stage_freq_hidden_dim, use_local=stage_use_local,
                fls_channelwise=stage_fls_channelwise,
                layer_scale_init_value=stage_layer_scale)
        )

    return nn.Sequential(*blocks)

class RCViT(nn.Module):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=[True, True, True, True], norm_layer=nn.BatchNorm2d, attn_bias=False,
                 act_layer=nn.GELU, num_classes=1000, drop_rate=0., drop_path_rate=0., fork_feat=False,
                 init_cfg=None, pretrained=None, distillation=True, use_fls_stages=None, fls_kernel_size=None,
                 freq_hidden_dim=16, use_local=True, fls_channelwise=True, layer_scale_init_value=1e-4, **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        if use_fls_stages is None:
            use_fls_stages = [False] * len(layers)
        if fls_kernel_size is None:
            fls_kernel_size = [7, 7, 9, 7][:len(layers)]

        mlp_ratios = _expand_stage_config(mlp_ratios, len(layers))
        use_fls_stages = _expand_stage_config(use_fls_stages, len(layers))
        fls_kernel_size = _expand_stage_config(fls_kernel_size, len(layers))
        freq_hidden_dim = _expand_stage_config(freq_hidden_dim, len(layers))
        use_local = _expand_stage_config(use_local, len(layers))
        fls_channelwise = _expand_stage_config(fls_channelwise, len(layers))
        layer_scale_init_value = _expand_stage_config(layer_scale_init_value, len(layers))

        assert len(layers) == len(embed_dims)
        assert len(mlp_ratios) == len(layers)
        assert len(use_fls_stages) == len(layers)
        assert len(fls_kernel_size) == len(layers)

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios, act_layer=act_layer,
                          attn_bias=attn_bias, drop=drop_rate, drop_path_rate=drop_path_rate,
                          use_fls_stages=use_fls_stages, fls_kernel_size=fls_kernel_size,
                          freq_hidden_dim=freq_hidden_dim, use_local=use_local,
                          fls_channelwise=fls_channelwise,
                          layer_scale_init_value=layer_scale_init_value)

            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=3, stride=2, padding=1, in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1], norm_layer=nn.BatchNorm2d)
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        pass

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out

# ======================================================================================================================

@register_model
def test1(**kwargs):
    model = RCViT(
        layers=[2, 2, 6, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[False, True, True, False],
        fls_kernel_size=[7, 7, 7, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        **kwargs)
    return model


@register_model
def test2(**kwargs):
    model = RCViT(
        layers=[2, 2, 8, 2], embed_dims=[48, 64, 128, 256], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[False, True, True, False],
        fls_kernel_size=[7, 7, 7, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        **kwargs)
    return model


@register_model
def test3(**kwargs):
    model = RCViT(
        layers=[2, 3, 12, 3], embed_dims=[64, 96, 192, 384], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[True, True, True, True],
        fls_kernel_size=[7, 9, 9, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        **kwargs)
    return model


@register_model
def test4(**kwargs):
    model = RCViT(
        layers=[2, 2, 8, 2], embed_dims=[48, 64, 128, 256], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[True, True, True, True],
        fls_kernel_size=[7, 7, 7, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        **kwargs)
    return model

@register_model
def test5(**kwargs):
    model = RCViT(
        layers=[2, 3, 12, 3], embed_dims=[64, 96, 144, 256], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[True, True, True, True],
        fls_kernel_size=[7, 9, 9, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        **kwargs)
    return model

@register_model
def test6(**kwargs):
    model = RCViT(
        layers=[2, 2, 6, 2], embed_dims=[40, 48, 96, 160], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[False, True, True, False],
        fls_kernel_size=[7, 7, 7, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        **kwargs)
    return model

@register_model
def test7(**kwargs):
    model = RCViT(
        layers=[2, 3, 12, 3], embed_dims=[64, 96, 144, 256], mlp_ratios=[2, 2, 3, 3], downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None,
        use_fls_stages=[True, True, True, True],
        fls_kernel_size=[7, 9, 9, 7],
        freq_hidden_dim=[4, 4, 8, 4],
        fls_channelwise=[True, True, True, True],
        use_local=False,
        **kwargs)
    return model

# ======================================================================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size=(1, 3, 224, 224)):
    from thop import profile
    dummy_input = torch.randn(input_size)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops, params


if __name__ == '__main__':
    net = test7()
    x = torch.rand((1, 3, 224, 224))
    out = net(x)

    if isinstance(out, tuple):
        print('Net Output Shapes: {}'.format([o.shape for o in out]))
    else:
        print('Net Output Shape: {}'.format(out.shape))

    print('Net Params: {:d}'.format(int(count_parameters(net))))

    print('Net FLOPs: {:.2f} M'.format(count_flops(net)[0] / 1e6))
    print('Net Params: {:.2f} M'.format(count_flops(net)[1] / 1e6))

