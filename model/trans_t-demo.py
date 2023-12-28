import os
import sys
object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)
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

from thop import profile,clever_format

class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores

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

class ModulatedGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        self.adj = adj

        self.adj2 = nn.Parameter(torch.ones_like(adj))        
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    
    def forward(self, input, is_train = False):

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        
        adj = self.adj.to(input.device) + self.adj2.to(input.device)
  
        adj = (adj.T + adj)/2
        
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)
        #32*2048*16
        return output

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv =  ModulatedGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

# ------------------------------#
# Positional Encoding的代码实现
# ------------------------------#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()

        # 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面的代码只是其中的一种实现方式；
        # 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        # pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        # 假设我的d_model是512，2i以步长2从0取到了512，那么i对应取值就是0,1,2...255
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，步长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，步长为2，其实代表的就是奇数位置

        # 下面这行代码之后，我们得到的pe形状是：[max_len * 1 * d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ----------------------------#
# PoswiseFeedForwardNet的实现
# ----------------------------#
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,in_channels=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=in_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class Model(nn.Module):
    def __init__(self, cfg, n_layers=2, d_model=512):
        super(Model, self).__init__()
        self.proj = nn.Linear(2048, d_model)
        self.trans = Transformer(n_layers, d_model, d_model*4, cfg.DATASET.SEQLEN)
        self.out_proj = nn.Linear(d_model, 2048)

        self.regressor = Regressor()

        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressor.load_state_dict(pretrained_dict, strict=False)
        
        self.trr = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1,padding=1, bias=False)
        self.adj16 = nn.Parameter(torch.ones(16, 16))
        self.gcn16 = ModulatedGraphConv(2048,2048,adj=self.adj16)
        self.pe = PositionalEncoding(d_model=2048)
        self.trr = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1,padding=1, bias=False)
        self.ffn16 = PoswiseFeedForwardNet(in_channels=16)
        self.layer_norm = nn.LayerNorm(2048)
        self.mid_frame = int(16/2)
        self.hidden_size = 2048

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False, J_regressor=None):
        x_trr = self.trr(x)
        y_cur = self.pe(x_trr.permute(1,0,2)).permute(1,0,2) + x_trr
        y_cur = self.gcn16(y_cur) + x_trr
        y_cur = self.ffn16(y_cur.permute(0,2,1)).permute(0,2,1)
        
        x = self.proj(x)     # B 16 512
        x = self.trans(x)    # B 16 512
        x = self.out_proj(x) # B 16 2048
        
        x=x+y_cur

        # output[0]['kp_3d']: B, 16, 49, 3    B, 1, 49, 3
        output = self.regressor(x, is_train=is_train, J_regressor=J_regressor)
        
        return output

if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.SEQLEN = 16

    batch_size = 64
    model = Model(cfg) # 模型初始化
    model = model.cuda()
    
    input_2d = torch.randn(64, 16, 2048).cuda()
    flops, params = profile(model, inputs=(input_2d, ))
    print('flops: ', flops, 'params: ', params)
    flops, params = clever_format([flops, params], "%.3f")
    print('ours:',flops, params)
    model.eval()
        
# 3DPW{'mpjpe': 83.14, 'mpjpe_pa': 51.59, 'accel_err': 8.06, 'mpvpe': 98.51}
# {'mpjpe': 97.01, 'mpjpe_pa': 62.84, 'accel_err': 9.45}
# MPII3D{'mpjpe': 95.86, 'mpjpe_pa': 61.71, 'accel_err': 9.2}
# {'mpjpe': 68.86, 'mpjpe_pa': 46.18, 'accel_err': 4.65}
# H36M{'mpjpe': 67.17, 'mpjpe_pa': 46.04, 'accel_err': 4.85}

# {'mpjpe': 81.54, 'mpjpe_pa': 51.94, 'accel_err': 7.57, 'mpvpe': 97.19}
# {'mpjpe': 82.0, 'mpjpe_pa': 51.43, 'accel_err': 8.09, 'mpvpe': 98.66}

