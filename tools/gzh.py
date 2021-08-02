"""
Description: gzh常用小工具

@author: GuZhouhong
@date: 2021/04/04
"""
from collections import defaultdict
import numpy, os
import json, torch


class dr:
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.poi_path = os.path.join(data_path, 'poi.json')
        self.hypo_path = os.path.join(data_path, 'hyponym.json')
        self.hyper_path = os.path.join(data_path, 'hypernym.json')

        self.concept_path = os.path.join(data_path, 'concept.txt')
        self.ori_triple_path = os.path.join(data_path, 'triples.data')
        self.triple_path = os.path.join(data_path, 'filter_triples.txt')

        self.spu_json = os.path.join(data_path, 'spu.json')
        self.syn_json = os.path.join(data_path, 'synonym.json')
        self.dish_txt = os.path.join(data_path, '100w_dish.txt')

        self.cate_txt = os.path.join(data_path, 'cate_id.txt')
        self.poi_spu_filter = os.path.join(data_path, 'poi_spu_filter_20.txt')


cache_dir = r"D:\公用数据\bert_语言模型"

helper = 0


def getContext(hyper, hypo, use_template):
    if not use_template:
        return '%s[SEP]%s。' % (hypo, hyper)
    else:
        return '%s是一种%s。' % (hypo, hyper)


def getContextList(hypo_lis, hyper_lis):
    context = []
    for i, j in zip(hypo_lis, hyper_lis):
        context.append(getContext(i, j))
    return context


def getMetrics(tp, fp, tn, fn):
    num = (tp + tn + fp + fn)
    acc = (tp + tn) / num
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * pre * recall / (pre + recall)
    return acc, pre, recall, f1


def getDataset(path):
    output_one = []
    output_zero = []
    f = open(path, encoding='utf-8')
    for line in f:
        a, b, c = line.strip().split('\t')
        if c == '1':
            output_one.append(b)
        if c == '0':
            output_zero.append(b)
    f.close()
    return output_one, output_zero


wwm_bert_model = r'D:\公用数据\tfhub\chinese_roberta_wwm_ext_L-12_H-768_A-12'
bert_model = r'D:\公用数据\bert_语言模型\chinese_L-12_H-768_A-12'
albert_model = r'D:\公用数据\tfhub\albert_base'
mt_bert = r'D:\公用数据\tfhub\meituan_L-12_H-768_A-12'

from pytorch_transformers import BertModel, BertTokenizer
import torch.nn as nn


class bert_encoder(nn.Module):
    def __init__(self, bert_name, device='cuda'):
        super(bert_encoder, self).__init__()
        if bert_name == 'mt_bert':
            bert_path = mt_bert
        elif bert_name == 'chinese-base-bert':
            bert_path = bert_model
        elif bert_name == 'wwm_bert':
            bert_path = wwm_bert_model
        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.to(device)
        self.device = device

    def forward(self, sentence, max_len=16):
        tokens = [self.tokenizer.tokenize(str(i)) for i in sentence]
        tensor = [self.tokenizer.convert_tokens_to_ids(t)[:max_len] + [0] * max(0, max_len - len(t)) for t in tokens]
        outputs = self.bert(torch.tensor(tensor).to(self.device))[1]
        return outputs.detach().cpu().numpy()


class TrieNode(object):
    def __init__(self, value=None):
        # 值
        self.value = value
        # fail指针
        self.fail = None
        # 尾标志：标志为i表示第i个模式串串尾，默认为0
        self.tail = 0
        # 子节点，{value:TrieNode}
        self.children = {}


class Trie(object):
    def __init__(self, words=[]):
        # 根节点
        self.root = TrieNode()
        # 模式串个数
        self.count = 0
        self.words = words
        for word in words:
            self.insert(word)
        self.ac_automation()

    def add_words(self, words):
        self.words.extend(words)
        for word in words:
            self.insert(word)
        self.ac_automation()

    def insert(self, word):
        """
        基操，插入一个字符串
        :param word: 字符串
        :return:
        """
        self.count += 1
        cur_node = self.root
        for char in word:
            if char not in cur_node.children:
                # 插入结点
                child = TrieNode(value=char)
                cur_node.children[char] = child
                cur_node = child
            else:
                cur_node = cur_node.children[char]
        cur_node.tail = self.count

    def ac_automation(self):
        """
        构建失败路径
        :return:
        """
        queue = [self.root]
        # BFS遍历字典树
        while len(queue):
            temp_node = queue[0]
            # 取出队首元素
            queue.remove(temp_node)
            for value in temp_node.children.values():
                # 根的子结点fail指向根自己
                if temp_node == self.root:
                    value.fail = self.root
                else:
                    # 转到fail指针
                    p = temp_node.fail
                    while p:
                        # 若结点值在该结点的子结点中，则将fail指向该结点的对应子结点
                        if value.value in p.children:
                            value.fail = p.children[value.value]
                            break
                        # 转到fail指针继续回溯
                        p = p.fail
                    # 若为None，表示当前结点值在之前都没出现过，则其fail指向根结点
                    if not p:
                        value.fail = self.root
                # 将当前结点的所有子结点加到队列中
                queue.append(value)

    def search_word(self, word:str):
        answers = self.search(word)
        for answer in answers:
            if answer == word:
                return True
        return False

    def search(self, text):
        """
        模式匹配
        :param self:
        :param text: 长文本
        :return:
        """
        p = self.root
        # 记录匹配起始位置下标
        start_index = 0
        # 成功匹配结果集
        rst = defaultdict(list)
        for i in range(len(text)):
            single_char = text[i]
            while single_char not in p.children and p is not self.root:
                p = p.fail
            # 有一点瑕疵，原因在于匹配子串的时候，若字符串中部分字符由两个匹配词组成，此时后一个词的前缀下标不会更新
            # 这是由于KMP算法本身导致的，目前与下文循环寻找所有匹配词存在冲突
            # 但是问题不大，因为其标记的位置均为匹配成功的字符
            if single_char in p.children and p is self.root:
                start_index = i
            # 若找到匹配成功的字符结点，则指向那个结点，否则指向根结点
            if single_char in p.children:
                p = p.children[single_char]
            else:
                start_index = i
                p = self.root
            temp = p
            while temp is not self.root:
                # 尾标志为0不处理，但是tail需要-1从而与敏感词字典下标一致
                # 循环原因在于，有些词本身只是另一个词的后缀，也需要辨识出来
                if temp.tail:
                    rst[self.words[temp.tail - 1]].append((start_index, i))
                temp = temp.fail
        return rst


def getLayer(hypo, layer_num):
    # 第一层
    keys = list(hypo.keys())
    concept = set(keys)
    # 第i+1层
    for i in range(layer_num - 1):
        for key in keys:
            if key in hypo:
                for entity in hypo[key]:
                    concept.add(entity)
        keys = concept
    return concept


def getAcc(labels, preds):
    pass


def toJson(dic, path):
    jsonData = json.dumps(dic, ensure_ascii=False, indent=4)
    f = open(path, 'w', encoding='utf-8')
    f.write(jsonData)
    f.close()


def readJson(path):
    f = open(path, encoding='utf-8')
    jsonData = json.load(f)
    f.close()
    return jsonData

# 获取所有子孙
def getDesc(word, lis: list, hypo):
    if word in hypo:
        for value in hypo[word]:
            if value in lis:
                continue
            else:
                lis.append(value)
                lis = getDesc(value, lis, hypo)
    return list(set(lis))

def getDummies(label, size=2):
    l = [0] * size
    l[label] = 1
    return l

