'''
@author: Zhouhong Gu
@date: 2021/07/26
@target: 双层图神经网络
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import config
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat=config.nfeat, nhid=config.nhid, nclass=1, dropout=config.dropout, layer_num=2):
        super(GCN, self).__init__()

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid))
        for i in range(layer_num - 2):
            self.gcs.append(GraphConvolution(nhid, nhid))
        self.gcs.append(GraphConvolution(nhid, nclass))
        self.fc = nn.Linear(nhid * 2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        '''
        :param x:
        :param adj:
        :return:
        '''
        for index, layer in enumerate(self.gcs):
            if index == len(self.gcs) - 1: break
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcs[-1](x, adj)
        return F.log_softmax(x, dim=1)

    def forward_2(self, x, adj):
        for index, layer in enumerate(self.gcs):
            if index == len(self.gcs) - 1: break
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        # 先用一种很low的方法
        new_x = []
        for index1, line1 in enumerate(adj):
            for index2, line2 in enumerate(line1):
                if line2 > 0:
                    new_x.append([x.detach().cpu().numpy()[index1], x.detach().cpu().numpy()[index2]])
        new_x = torch.tensor(new_x)
        new_x = new_x.reshape(new_x.shape[0],-1)
        x = self.fc(new_x)
        return F.sigmoid(x)

    def get_embedding(self, x, adj):
        for index, layer in enumerate(self.gcs):
            if index == len(self.gcs) - 1: break
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return x


if __name__ == '__main__':
    nfeat = 768
    nhid = 768
    nclass = 1
    dropout = 0.4
    gcn = GCN(nfeat, nhid, nclass, dropout)
    x = torch.randn((20, 768))
    adj = torch.randn((20, 20))
    adj[adj > 0.5] = 1
    print(adj.shape)
    gcn.eval()
    print(gcn.forward_2(x, adj).shape)
