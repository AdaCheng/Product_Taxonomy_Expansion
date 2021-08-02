'''
@author: Zhouhong Gu
@date: 2021/07/26
@target: 只使用BERT
'''

import torch
from tools import gzh
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_transformers import BertTokenizer, BertModel
from model.GCN.layers import GraphConvolution
import random
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
use_cl = config.use_cl  # 对比学习

mt_bert = r'.\pre-trainLM\meituan_L-12_H-768_A-12'
bert_model = r'.\pre-trainLM\chinese_L-12_H-768_A-12'
bert_pre = config.bert_pre
assert bert_pre in ['chinese-base-bert', 'mt_bert']
use_pos_dim = config.use_pos_emb
pos_emb_size = config.pos_emb_size
device = 'cuda'


class bertGCNClassifier(nn.Module):
    def __init__(self, nfeat=config.nfeat, nhid=config.nhid, nclass=1, dropout=config.dropout, layer_num=2):
        super(bertGCNClassifier, self).__init__()
        self.device = 'cuda'
        self.initgcn(nfeat=nfeat, nhid=nhid, nclass=1, dropout=dropout, layer_num=2)
        self.initbert()
        if not use_pos_dim:
            self.fc = nn.Linear(768 + nhid * 2, nclass)
        if use_pos_dim:
            self.fc = nn.Linear(768 + nhid * 2, nclass)

    def initbert(self):
        if bert_pre == 'chinese-base-bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)
            self.model = BertModel.from_pretrained(bert_model)
        elif bert_pre == 'mt_bert':
            self.tokenizer = BertTokenizer.from_pretrained(mt_bert)
            self.model = BertModel.from_pretrained(mt_bert)
        self.model.to(self.device)
        # model返回一个768维度的向量

    def initgcn(self, nfeat=config.nfeat, nhid=config.nhid, nclass=1, dropout=config.dropout, layer_num=2):
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid))
        for i in range(layer_num - 2):
            self.gcs.append(GraphConvolution(nhid, nhid))
        self.gcs.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout
        # gcn返回一排768维的向量

    def forward(self, x, adj, word2id, batch_size=32, max_len=16):
        for index, layer in enumerate(self.gcs):
            if index == len(self.gcs) - 1: break
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        # 先用一种很low的方法并且很耗时的方法，后续考虑能否直接使用矩阵计算得到
        new_x = []
        new_x2 = []
        sentence = []
        sentence2 = []
        for index1, line1 in enumerate(adj):
            for index2, line2 in enumerate(line1):
                if line2 > 0:
                    if not use_cl:
                        new_x.append([x.detach().cpu().numpy()[index1], x.detach().cpu().numpy()[index2]])
                    if use_cl:
                        new_x.append(
                            [(1 - lr) * x.detach().cpu().numpy()[index1] + lr * x.detach().cpu().numpy()[index2],
                             (1 - lr) * x.detach().cpu().numpy()[index2] + lr * x.detach().cpu().numpy()[index1]])
                    sentence.append(config.template([word2id[index1], word2id[index2]]))
                else:
                    if not use_cl:
                        new_x2.append([x.detach().cpu().numpy()[index1], x.detach().cpu().numpy()[index2]])
                    if use_cl:
                        new_x.append(
                            [(1 + lr) * x.detach().cpu().numpy()[index1] - lr * x.detach().cpu().numpy()[index2],
                             (1 + lr) * x.detach().cpu().numpy()[index2] - lr * x.detach().cpu().numpy()[index1]])
                    sentence2.append(config.template([word2id[index1], word2id[index2]]))
        new_x = torch.tensor(new_x)
        new_x = new_x.reshape(new_x.shape[0], -1)

        idx = random.sample(list(range(len(sentence2))), k=len(sentence))
        sentence2 = [sentence2[i] for i in idx]
        new_x2 = torch.tensor([new_x2[i] for i in idx])
        new_x2 = new_x2.reshape(new_x2.shape[0], -1)

        new_x = torch.cat((new_x, new_x2), dim=0)
        sentence.extend(sentence2)

        del (new_x2)
        del (sentence2)

        idx = random.sample(list(range(len(sentence))), k=len(sentence))

        new_x = new_x[idx][:batch_size]
        sentence = np.array(sentence)[idx][:batch_size]
        x = self.model(torch.tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i))[:max_len] + [
            0] * max(0, max_len - len(self.tokenizer.tokenize(i))) for i in sentence]).to(self.device))[1]
        x = self.fc(torch.cat((torch.tensor(new_x), torch.tensor(x.detach().cpu().numpy())), dim=1))
        return F.sigmoid(x)


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
    return torch.tensor(features), torch.tensor(adj), childs, ances, id2word


def trainBERTGCN(model, optimizer, creiterion, features, adj, childs, word2id):
    model.train()
    optimizer.zero_grad()
    outputs = model(features, adj, word2id)
    epoch_loss = creiterion(outputs.reshape(-1), torch.FloatTensor(sorted(childs, reverse=True)[:outputs.shape[0]]))
    epoch_loss.backward()
    optimizer.step()

    return epoch_loss


def predictBERTGCN(model, features, adj, childs, ances, word2id):
    model.eval()
    outputs = model(features, adj, word2id).reshape(-1)
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

    from tools import gzh
    acc, pre, recall, f1 = gzh.getMetrics(tp, fp, tn, fn)
    acca, prea, recalla, f1a = gzh.getMetrics(tpa, fpa, tna, fna)
    return acc, f1, f1a
