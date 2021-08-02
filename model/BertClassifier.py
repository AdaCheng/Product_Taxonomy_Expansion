'''
@author: Zhouhong Gu
@date: 2021/07/26
@target: 只使用BERT
'''

import random
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_transformers import BertTokenizer, BertModel

import config

mt_bert = r'.\pre-trainLM\meituan_L-12_H-768_A-12'
bert_model = r'.\pre-trainLM\chinese_L-12_H-768_A-12'
bert_pre = config.bert_pre
assert bert_pre in ['chinese-base-bert', 'mt_bert']


class bertClassifier(nn.Module):
    def __init__(self, device='cuda'):
        super(bertClassifier, self).__init__()
        if bert_pre == 'chinese-base-bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)
            self.model = BertModel.from_pretrained(bert_model)
        elif bert_pre == 'mt_bert':
            self.tokenizer = BertTokenizer.from_pretrained(mt_bert)
            self.model = BertModel.from_pretrained(mt_bert)
        self.linear = nn.Linear(768, 1)
        self.device = device
        self.model.to(device)
        self.linear.to(device)

    def forward(self, sentence, max_len=16):
        x = self.model(torch.tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i))[:max_len] + [
            0] * max(0, max_len - len(self.tokenizer.tokenize(i))) for i in sentence]).to(self.device))[1]
        x = self.linear(x)
        return F.sigmoid(x)


def getContext(x):
    return config.template(x)


class myDataset(Dataset):
    def __init__(self, data):
        self.len = len(data)

        self.data = np.array([getContext(x) for x in data])
        self.child = np.array([float(x[2]) for x in data])
        self.ance = np.array([float(x[3]) for x in data])

    def __getitem__(self, index):
        sample = self.data[index], self.child[index], self.ance[index]
        return sample

    def __len__(self):
        return self.len


def trainClassifier(train, model, criterion, optimizer, batch_size, save_model):
    global myDataset
    model.train()
    dataset = myDataset(train.copy())
    # train_loader = DataLoader(dataset=myDataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    epoch_loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x, childs, _ in train_loader:
        outputs = model(x)
        for l, o in zip(childs, outputs):
            if l == 1:
                if o > 0.5:
                    tp += 1
                else:
                    fn += 1
            else:
                if o > 0.5:
                    fp += 1
                else:
                    tn += 1
        labels = torch.DoubleTensor(childs).to('cuda').reshape(-1, 1)
        loss = criterion(outputs, labels.float())
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if save_model:
        torch.save(model.state_dict(), './model/pkls/BertClassifier.pkl')
    num = (tp + tn + fp + fn)
    acc = (tp + tn) / num
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * pre * recall / (pre + recall)
    aa = '训练集数据：Edge 准确率%.4f，f1:%.4f' % (acc, f1)
    return epoch_loss, aa


def predictClassifier(test, model, batch_size):
    global myDataset
    model.eval()
    dataset = myDataset(test.copy())
    # train_loader = DataLoader(dataset=myDataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tpa = 0
    tna = 0
    fpa = 0
    fna = 0
    for x, childs, ances in test_loader:
        outputs = model(x)
        for output, label in zip(outputs, childs):
            if int(label) == 1:
                if output > 0.5:
                    tp += 1
                else:
                    fn += 1
            else:
                if output > 0.5:
                    fp += 1
                else:
                    tn += 1
        for output, label in zip(outputs, ances):
            if int(label) == 1:
                if output > 0.5:
                    tpa += 1
                else:
                    fna += 1
            else:
                if output > 0.5:
                    fpa += 1
                else:
                    tna += 1

    num = (tp + tn + fp + fn)
    acc = (tp + tn) / num
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * pre * recall / (pre + recall)
    numa = (tpa + tna + fpa + fna)
    acca = (tpa + tna) / numa
    prea = tpa / (tpa + fpa)
    recalla = tpa / (tpa + fna)
    f1a = 2 * prea * recalla / (prea + recalla)
    return (acc, f1), (acca, f1a)


if __name__ == '__main__':
    from tools import gzh
    import config, time

    use_template = config.use_template
    batch_size = 256

    match = gzh.readJson('../result/userGraph.json')
    data = []
    for key in tqdm.tqdm(match):
        for value in match[key]:
            data.append([key, value])
    del (match)
    bc = bertClassifier()
    bc.load_state_dict(torch.load('./BertClassifier.pkl'))

    batch_dat = []
    import time

    f = open('../result/%s_hypo-hyper.txt' % ''.join([str(i) for i in time.localtime()]), 'w', encoding='utf-8')
    index = 0
    start = time.time()
    while (data):
        index += 1
        if index % 10 == 0:
            print('剩余:%d个，耗时:%.4f' % (len(data), time.time() - start))
        batch_dat.extend([gzh.getContext(i, j, use_template) for i, j in data[:batch_size]])
        data = data[batch_size:]
        outputs = bc(batch_dat)
        for output, dat in zip(outputs, batch_dat):
            if output[1] > output[0]:
                f.write(str(dat) + str(output) + '\n')
            else:
                f.write('\t\t错误：' + str(dat) + str(output) + '\n')
        batch_dat = []
