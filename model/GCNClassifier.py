'''
@author: Zhouhong Gu
@date: 2021/07/26
@target: GCN的训练过程
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN.layers import GraphConvolution
from model.GCN.utils import load_data, accuracy
import random
from sklearn.metrics import f1_score, accuracy_score
import config

# Training settings
GCN_layer = config.GCN_layer  # GCN层数
nfeat = config.nfeat  # 特征层
nhid = config.nhid  # 隐藏层
nclass = config.nclass  # 分类结果
dropout = config.dropout  # dropout
epochs = config.epochs  # epoch
lr = config.lr  # learning rate
use_cuda = config.use_cuda  # 使用cuda
fastmode = config.fastmode  # 需不需要验证


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
        # 先用一种很low的方法并且很耗时的方法，后续考虑能否直接使用矩阵计算得到
        new_x = []
        for index1, line1 in enumerate(adj):
            for index2, line2 in enumerate(line1):
                if line2 > 0:
                    new_x.append([x.detach().cpu().numpy()[index1], x.detach().cpu().numpy()[index2]])
        new_x = torch.tensor(new_x)
        new_x = new_x.reshape(new_x.shape[0], -1)
        # print('new_x shape', new_x.shape)
        x = self.fc(new_x)
        return F.sigmoid(x)

    def get_embedding(self, x, adj):
        for index, layer in enumerate(self.gcs):
            if index == len(self.gcs) - 1: break
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return x


def getFeatureAdj(data):
    from tools import gzh
    bert_pre = config.bert_pre
    encoder = gzh.bert_encoder(bert_pre)

    words = set([item for sublist in [[i[0], i[1]] for i in data] for item in sublist])

    word2id = {i: j for j, i in enumerate(words)}
    id2word = {word2id[i]: i for i in word2id}

    temp = [[word2id[i[0]], word2id[i[1]]] for i in data]
    graph = {}
    for a, b in temp:
        li = graph.get(a, set())
        li.add(b)
        graph[a] = li
    del (temp)

    adj = []
    for d in id2word:
        adj.append([0.] * len(word2id))
        for dd in graph.get(d, []):
            adj[-1][dd] = 1.
    del (graph)
    print('adj shape', np.array(adj).shape)

    features = []
    for word in word2id:
        features.append(encoder([word])[0])
    print('feature shape', np.array(features).shape)

    childs = [float(i[2]) for i in data]
    ances = [float(i[3]) for i in data]
    return features, adj, childs, ances


def trainGCN(model, optimizer, creiterion, features, adj, childs):
    model.train()
    optimizer.zero_grad()
    outputs = model.forward_2(torch.tensor(features), torch.FloatTensor(adj))
    epoch_loss = creiterion(outputs.reshape(-1), torch.FloatTensor(sorted(childs, reverse=True)[:outputs.shape[0]]))
    epoch_loss.backward()
    optimizer.step()

    return epoch_loss


def predictGCN(model, features, adj, childs, ances):
    model.eval()
    outputs = model.forward_2(torch.tensor(features), torch.FloatTensor(adj)).reshape(-1)
    answers, labels, labelsa = outputs.cpu().detach().numpy(), torch.FloatTensor(
        sorted(childs, reverse=True)[:outputs.shape[0]]).cpu().detach().numpy(), torch.FloatTensor(
        sorted(ances, reverse=True)[:outputs.shape[0]]).cpu().detach().numpy()
    tp, fp, tn, fn = 0, 0, 0, 0
    tpa, fpa, tna, fna = 0, 0, 0, 0
    for a, l, la in zip(answers, labels, labelsa):
        if a > 0.5:
            if l == 1:
                tp += 1
            else:
                fp += 1
            if la == 1:
                tpa += 1
            else:
                fpa += 1
        else:
            if l == 0:
                tn += 1
            else:
                fn += 1
            if la == 0:
                tna += 1
            else:
                fna += 1

    # # acc = accuracy_score(outputs.cpu().detach().numpy(), torch.FloatTensor(sorted(childs, reverse=True)[:outputs.shape[0]]).cpu().detach().numpy())
    # f1 = f1_score(outputs.cpu().detach().numpy(),
    #               torch.FloatTensor(sorted(childs, reverse=True)[:outputs.shape[0]]).cpu().detach().numpy())
    # f1a = f1_score(outputs.cpu().detach().numpy(),
    #                torch.FloatTensor(sorted(ances, reverse=True)[:outputs.shape[0]]).cpu().detach().numpy())
    from tools import gzh
    acc, pre, recall, f1 = gzh.getMetrics(tp, fp, tn, fn)
    acca, prea, recalla, f1a = gzh.getMetrics(tpa, fpa, tna, fna)
    return acc, f1, f1a
