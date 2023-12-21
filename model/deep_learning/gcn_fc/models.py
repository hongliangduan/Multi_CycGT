import numpy as np
import torch
from torch import  nn
import torch.nn.functional as F
import copy
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor


class Model_TGCN(nn.Module):
    def __init__(self):
        super(Model_TGCN,self).__init__()
        self.num_fc = nn.Linear(103, 128)
        self.nrom = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(40+128, 1)
        self.sig = nn.Sigmoid()

    def forward(self,x_g,x_l):
        gcn_i=x_g.to("cuda")
        x_l=x_l.to("cpu")
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out).to("cuda")
        p = torch.cat([gcn_i,num_out],dim=-1).to("cpu")
        out = self.fc1(p)
        out = self.sig(out)
        return out
