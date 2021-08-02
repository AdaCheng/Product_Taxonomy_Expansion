'''
@author: Zhouhong Gu
@date: 2021/07/26
@target: GCN的训练过程
'''

import time
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN
from model.GCN import config
import torch
import numpy as np

# Training settings
GCN_layer = config.GCN_layer  # GCN层数
nfeat = config.nfeat  # 特征层
nhid = config.nhid  # 隐藏层
nclass = config.nclass  # 分类结果
dropout = config.dropout  # dropout
epochs = config.epochs  # epoch
lr = config.lr  # learning rate
use_cuda = config.use_cuda  # 使用cuda
weight_decay = config.weight_decay  # 权重衰减
fastmode = config.fastmode  # 需不需要验证

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=nhid,
            nclass=labels.max().item() + 1,
            dropout=dropout,
            layer_num=GCN_layer)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

if use_cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(model.get_embedding(features, adj).shape)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
