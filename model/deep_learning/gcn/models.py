import numpy as np
import torch
from torch import  nn
import torch.nn.functional as F
import copy
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor


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


class Model_TGCN(nn.Module):
    def __init__(self):
        super(Model_TGCN,self).__init__()
        self.num_fc = nn.Linear(103, 128)
        #归一化
        self.nrom = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(40, config.num_classes)

        self.sig = nn.Sigmoid()

    def forward(self,x_g):
        gcn_i = x_g.to("cuda")
        gcn_i = gcn_i.to("cpu")
        # out=x_t.to("cuda")
        # x_l=x_l.to("cpu")
        #全连接
        # num_out = self.num_fc(x_l)
        #归一化
        # num_out = self.nrom(num_out).to("cuda")
        #三者拼接
        # p = torch.cat([gcn_i,num_out],dim=-1).to("cpu")
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        #分类
        out = self.fc1(gcn_i)
        #激活
        out = self.sig(out)
        return out
