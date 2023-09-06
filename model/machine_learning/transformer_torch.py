import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
from torch import  nn
import torch.nn.functional as F
import copy

def get_position_encoding(seq_len, embed):
    pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])  # 公式实现
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe

pe = get_position_encoding(100,100)
sns.heatmap(pe)
plt.xlabel('emb')
plt.ylabel('seq_len')
plt.show()
# plt.clf()

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])   # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])   # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to("cuda")
        out = self.dropout(out)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)   # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape 回原来的形状
        out = self.fc(context)   # 全连接
        out = self.dropout(out)
        out = out + x      # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class ConfigTrans(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 1                    # 类别数
        self.num_epochs = 100                # epoch数
        self.batch_size = 12             # mini-batch大小
        self.pad_size = 1                    # 每句话处理成的长度(短填长切)，这个根据自己的数据集而定
        self.learning_rate = 0.001                    # 学习率
        self.embed = 128         # 字向量维度
        self.dim_model = 128      # 需要与embed一样
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 8       # 多头注意力，注意需要整除
        self.num_encoder = 2    # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM
config = ConfigTrans()
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        #分类器
        self.fc1 = nn.Linear(2048+40+128, config.num_classes)
        #全连接
        self.num_fc = nn.Linear(103, 128)
        #归一化
        self.nrom = nn.BatchNorm1d(128)
        #激活
        self.sig = nn.Sigmoid()
    def forward(self, x,gcn_i,num):
        #Transformer
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        #GCN的输入
        gcn_i=gcn_i.to("cuda")
        out=out.to("cuda")
        #全连接
        num_out = self.num_fc(num)
        #归一化
        num_out = self.nrom(num_out)
        #三者拼接
        p = torch.cat([out,gcn_i,num_out],dim=-1)
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        #分类
        out = self.fc1(p)
        #激活
        out = self.sig(out)
        return out

import pandas as pd
# for i in range(1,11):
#
#     X_train=pd.read_csv('../../data/data_splitClassifier_600/X_train{}.csv'.format(i))
#     X_test=pd.read_csv('../../data/data_splitClassifier_600/X_test{}.csv'.format(i))
#     X_val=pd.read_csv('../../data/data_splitClassifier_600/X_val{}.csv'.format(i))
#     y_train=pd.read_csv('../../data/data_splitClassifier_600/y_train{}.csv'.format(i))
#     y_test=pd.read_csv('../../data/data_splitClassifier_600/y_test{}.csv'.format(i))
#     y_val=pd.read_csv('../../data/data_splitClassifier_600/y_val{}.csv'.format(i))
#     # 显示各个集合的数据量
#     print('第{}次交叉验证'.format(i))
#     print(f"Training set: {X_train.shape[0]} samples")
#     print(f"Validation set: {X_val.shape[0]} samples")
#     print(f"Testing set: {X_test.shape[0]} samples")
#
#     # model = TransformerClassifier(num_classes=2)
#     model = Transformer()
#     model.to("cpu")
#     print(model(torch.tensor([1]*50).cpu()))
