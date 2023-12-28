import os
import sys
sys.path.append("..")
import torch
import os.path as osp
import torch.nn as nn
from common.arguments import BASE_DATA_DIR
from einops import rearrange
from IPython import embed
from model.utils.spin import Regressor
import math
from functools import partial
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth=4, embed_dim=512, mlp_hidden_dim=2048, length=16, h=8):
        super().__init__()
        drop_rate = 0.1
        drop_path_rate = 0.2
        attn_drop_rate = 0.

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class Model(nn.Module):
    def __init__(self, cfg, n_layers=2, d_model=512):
        super(Model, self).__init__()
        self.proj = nn.Linear(2048, d_model)
        self.trans = Transformer(n_layers, d_model, d_model*4, cfg.DATASET.SEQLEN)
        self.out_proj = nn.Linear(d_model, 2048)

        self.regressor = Regressor()

        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressor.load_state_dict(pretrained_dict, strict=False)

    def forward(self, x, is_train=False, J_regressor=None):
        x = self.proj(x)     # B 16 512
        x = self.trans(x)    # B 16 512
        x = self.out_proj(x) # B 16 2048

        # output[0]['kp_3d']: B, 16, 49, 3    B, 1, 49, 3
        output = self.regressor(x, is_train=is_train, J_regressor=J_regressor)
        
        return output

if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.SEQLEN = 16

    os.chdir('/home/bird/bird/Pose/VideoBody')

    model = Model(cfg)
    model.eval()

    model_params = 0
    for parameter in model.parameters(cfg):
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/ 1000000)

    inputs = torch.rand(64, 16, 2048)

    with torch.no_grad():
        print(inputs.shape, 1)
        output = model(inputs)
        print(output[0]['kp_3d'].shape, 2)

