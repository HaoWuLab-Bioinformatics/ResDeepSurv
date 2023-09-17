
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
from itertools import product
import torch.nn.functional as F
from torch import random
from torch.backends import cudnn

device = torch.device('cuda:3')



class Regularization(object):  
    def __init__(self, order, weight_decay):  # 初始化

        super(Regularization, self).__init__()
        self.order = order  
        self.weight_decay = weight_decay 

    def __call__(self, model):  
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim, activation, num_heads):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.activation = eval('nn.{}()'.format(activation))
        self.num_heads = num_heads

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.activation(self.fc2(x))
        alpha = F.softmax(x, dim=1)

        
        alpha = alpha.view(alpha.size(0), self.num_heads, -1)
        x = x.view(x.size(0), self.num_heads, -1)

 
        weighted_input = torch.mul(alpha, x)  # element-wise multiplication

        output = torch.sum(weighted_input, dim=1)

        return output

’‘’

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim, activation):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)

        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.activation = eval('nn.{}()'.format(activation))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.activation(self.fc2(x))
        alpha = F.softmax(x, dim=1)
        return alpha

‘’‘
class ResDeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self.attention = config['attention'] 
        self.attn_dim = config['attn_dim'] 
        self.attn_activation = config['attn_activation'] 
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        seed = 3045  # 宇宙答案
        random.seed()
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = True
        device = torch.device("cuda:3")
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None:
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
 #           if self.norm:
#                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            residual_block = []
            for j in range(2):
                residual_block.append(eval('nn.{}()'.format(self.activation)))
                residual_block.append(nn.Linear(self.dims[i + 1], self.dims[i + 1]))
                if self.norm:
                    residual_block.append(nn.BatchNorm1d(self.dims[i + 1]))
            layers.append(eval('nn.{}()'.format(self.activation)))
            residual_block = nn.Sequential(*residual_block)
            layers.append(nn.Sequential(
                nn.BatchNorm1d(self.dims[i + 1]),
                residual_block
            ))
            if self.attention and i < len(self.dims)-2:
                attn = Attention(self.dims[i+1], int(self.dims[i+1]),int(self.attn_dim), self.attn_activation)

                layers.append(attn)


        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):

class NegativeLogLikelihood(nn.Module):
    # 负对数似然函数
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask.to(device)
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0).to(device)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss