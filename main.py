'''
@author: Zhouhong Gu
@date: 2021/07/29
@target: 调用训练和验证过程
'''

from model.GCNClassifier import GCN, trainGCN, predictGCN
from model.BertClassifier import trainClassifier, bertClassifier, predictClassifier
from model.BertFeatureClassifier import bertGCNClassifier, trainBERTGCN, predictBERTGCN, getFeatureAdj
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import config

use_model = config.use_model

# 深度学习的超参数
lr = config.lr
train_batch_size = config.train_batch_size
test_batch_size = config.test_batch_size
epochs = config.epochs
save_model = True
datapath = './dataset'

# 获得训练集
train = [i.strip().split('\t') for i in open(os.path.join(datapath, 'train.txt'), encoding='utf-8').readlines()]
# 获得验证集
valid = [i.strip().split('\t') for i in open(os.path.join(datapath, 'valid.txt'), encoding='utf-8').readlines()]
# 获得测试集
test = [i.strip().split('\t') for i in open(os.path.join(datapath, 'test.txt'), encoding='utf-8').readlines()]

if use_model == 'GCN':
    model = GCN()
    print('\ntrain')
    features, adj, childs, ances = getFeatureAdj(train + valid + test)
    print('\nvalid')
    features_val, adj_val, childs_val, ances_val = getFeatureAdj(valid)
    print('\ntest')
    features_test, adj_test, childs_test, ances_test = getFeatureAdj(test)
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

if use_model == 'BERT':
    model = bertClassifier()
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

if use_model == 'BERT+GCN':
    model = bertGCNClassifier()
    print('\ntrain')
    features, adj, childs, ances, id2word_train = getFeatureAdj(train + valid + test)
    print('\nvalid')
    features_val, adj_val, childs_val, ances_val, id2word_valid = getFeatureAdj(valid)
    print('\ntest')
    features_test, adj_test, childs_test, ances_test, id2word_test = getFeatureAdj(test)
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []
for _ in range(epochs):
    if use_model == 'GCN':
        epoch_loss = trainGCN(model, optimizer, criterion, features, adj, childs)
        acc, f1, f1a = predictGCN(model, features_val, adj_val, childs_val, ances_val)
        print('验证集准确率:%.4f，edge f1:%.4f，ancestor f1:%.4f' % (acc, f1, f1a))
        losses.append(epoch_loss)
    if use_model == 'BERT':
        epoch_loss, aa = trainClassifier(train, model, criterion, optimizer, train_batch_size, save_model)
        (acc, f1), (acca, f1a) = predictClassifier(train, model, test_batch_size)
        print('验证集准确率:%.4f，edge f1:%.4f，ancestor f1:%.4f' % (acc, f1, f1a))
        losses.append(epoch_loss)
    if use_model == 'BERT+GCN':
        epoch_loss = trainBERTGCN(model, optimizer, criterion, features, adj, childs, id2word_train)
        acc, f1, f1a = predictBERTGCN(model, features_val, adj_val, childs_val, ances_val, id2word_valid)
        print('验证集准确率:%.4f，edge f1:%.4f，ancestor f1:%.4f' % (acc, f1, f1a))
        losses.append(epoch_loss)

plt.plot(losses)
plt.show()

del (train)
del (valid)
if use_model == 'GCN':
    acc, f1, f1a = predictGCN(model, features_test, adj_test, childs_test, ances_test)
if use_model == 'BERT':
    (acc, f1), (acca, f1a) = predictClassifier(test, model, test_batch_size)
if use_model == 'BERT+GCN':
    acc, f1, f1a = predictBERTGCN(model, features_test, adj_test, childs_test, ances_test, id2word_test)
print('\n\n测试集准确率:%.4f，edge f1:%.4f，ancestor f1:%.4f' % (acc, f1, f1a))
