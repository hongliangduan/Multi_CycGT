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
        self.fc1 = nn.Linear(40, 1)
        self.sig = nn.Sigmoid()

    def forward(self,x_g):
        gcn_i = x_g.to("cuda")
        gcn_i = gcn_i.to("cpu")
        out = self.fc1(gcn_i)
        out = self.sig(out)
        return out
