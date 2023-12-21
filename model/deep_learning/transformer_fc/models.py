import numpy as np
import pandas as pd
import torch
from torch import  nn
import torch.nn.functional as F
import copy
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

batch_size= 256



class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
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
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

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
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape
        # context = torch.squeeze(context,dim=1)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
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
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
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

    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 1
        self.num_epochs = 100
        self.batch_size = 12
        self.pad_size = 1
        self.learning_rate = 0.001
        self.embed = 128
        self.dim_model = 128
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 2

config = ConfigTrans()

class Transformer_test(nn.Module):
    def __init__(self):
        super(Transformer_test, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])
        self.before_list = []
        self.after_list = []

    def forward(self, x):
        #Transformer
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)

        return out

class Model_TGCN(nn.Module):
    def __init__(self):
        super(Model_TGCN,self).__init__()
        self.num_fc = nn.Linear(103, 128)
        self.nrom = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(batch_size*128+128, config.num_classes)
        self.sig = nn.Sigmoid()
    def forward(self,x_t,x_l):
        out=x_t.to("cuda")
        x_l=x_l.to("cpu")
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out).to("cuda")
        p = torch.cat([out,num_out],dim=-1).to("cpu")
        out = self.fc1(p)
        out = self.sig(out)
        return out
