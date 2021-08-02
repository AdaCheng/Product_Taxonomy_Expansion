"""
@author: Zhouhong Gu
@idea:  v1. 生成全量训练集
            1. 先找到所有taxonomy里面的非中心词的上下位关系
            2. 根据找到的上下位关系划分训练集和测试集
            3. 根据训练集的个数，加入3/7的有中心词的上下位关系
            4. 测试集中，加入颠倒的上下位关系
            5. 测试集中，加入ancestor关系
            6. 测试集中，用中心词补足和训练集1：5的正例比例
            7. 测试集中，negtive sample,补足1：5的正负比例
        v2. 主要用于捞非中心词
            1. 搜素所有非中心词的例子
            2. 搜素user-graph和existing-taxonomy中共现的下位词
"""

import gzh, os, random, tqdm, time

neg_size = 1  # 负样本与正样本比例
head_to_nohead = [3, 7]
shuffle_size = 0.5  # 逆置负样本比例
train_rate = 0.6  # 训练集比例
val_rate = 0.2  # 验证集比例
test_rate = 0.2  # 测试集比例

tree = ['点心', '熟食', '水果']


def getdataset(data_path, dataset_path):
    '''

    :param data_path:
    :param dataset_path:
    :return: 上位词，下位词，下位词是否是上位词的孩子，下位词是否是上位词的子孙
    '''
    dr = gzh.dr(data_path)
    hypo = gzh.readJson(dr.hypo_path)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    desc = [item for sublist in [gzh.getDesc(i, [], hypo) for i in tree] for item in sublist]

    all_pairs = [[i, j, 1, 1] for i in desc if i in hypo for j in hypo[i]]
    all_pairs = random.sample(all_pairs, k=len(all_pairs))
    no_head_pairs = [i for i in all_pairs if not i[1].endswith(i[0])]
    head_pairs = [i for i in all_pairs if i[1].endswith(i[0])]

    head_size = int(len(no_head_pairs) * head_to_nohead[0] / sum(head_to_nohead))
    head_pairs = random.sample(head_pairs, k=head_size)

    # no head
    train_size = int(len(no_head_pairs) * train_rate)
    val_size = int(len(no_head_pairs) * val_rate)
    # test_size = int(len(no_head_pairs) * test_rate)

    trainData = no_head_pairs[:train_size]
    valData = no_head_pairs[train_size:train_size + val_size]
    testData = no_head_pairs[train_size + val_size:]

    print('无中心词的上下位关系总共找到%d个' % len(no_head_pairs))
    print('加完无中心词的正例，训练集个数：%d，验证集个数：%d，测试集个数：%d' % (len(trainData), len(valData), len(testData)))

    # head
    train_size = int(len(head_pairs) * train_rate)
    val_size = int(len(head_pairs) * val_rate)
    # test_size = int(len(head_pairs) * test_rate)

    trainData.extend(head_pairs[:train_size])
    valData.extend(head_pairs[train_size:train_size + val_size])
    testData.extend(head_pairs[train_size + val_size:])

    print('加完有中心词的正例，训练集个数：%d，验证集个数：%d，测试集个数：%d' % (len(trainData), len(valData), len(testData)))

    values = [i[1] for i in all_pairs]
    # negtive sample
    neg_dataset = []
    for dataset in [no_head_pairs, head_pairs]:
        for key in dataset:
            i = random.random()
            if i > 0.5:
                neg_dataset.append([key[1], key[0], 0, 0])
            else:
                neg_dataset.append([key[0], random.sample([i for i in values if i not in hypo[key[0]]], k=1)[0], 0, 0])
    # 计算负样本中的祖先关系
    for index, value in enumerate(tqdm.tqdm(neg_dataset)):
        # 这步有点耗时，后续可能考虑优化？
        desc = gzh.getDesc(value[0], [], hypo)
        if value[1] in desc:
            neg_dataset[index][-1] = 1

    train_size, val_size = len(trainData), len(valData)
    trainData.extend(neg_dataset[:train_size])
    valData.extend(neg_dataset[train_size:train_size + val_size])
    testData.extend(neg_dataset[train_size + val_size:])
    # print('总共获得了%d个负例' % len(neg_dataset))
    print('加完负例，训练集个数：%d，验证集个数：%d，测试集个数：%d' % (len(trainData), len(valData), len(testData)))

    # 写入文件
    f = open(os.path.join(dataset_path, 'train.txt'), 'w', encoding='utf-8')
    trainData = sorted(trainData, key=lambda x: x[0])
    for a, b, c, d in trainData:
        f.write(str(a) + '\t' + str(b) + '\t' + str(c) + '\t' + str(d) + '\n')

    f = open(os.path.join(dataset_path, 'valid.txt'), 'w', encoding='utf-8')
    valData = sorted(valData, key=lambda x: x[0])
    for a, b, c, d in valData:
        f.write(str(a) + '\t' + str(b) + '\t' + str(c) + '\t' + str(d) + '\n')

    f = open(os.path.join(dataset_path, 'test.txt'), 'w', encoding='utf-8')
    testData = sorted(testData, key=lambda x: x[0])
    for a, b, c, d in testData:
        f.write(str(a) + '\t' + str(b) + '\t' + str(c) + '\t' + str(d) + '\n')
    f.close()

    # --------------------------------------------------------------------------- #
    # 统计字符
    char2id = {}

    # 统计词语
    concept2id = {}

    for data in [trainData, valData, testData]:
        for w1, w2, label, _ in data:
            if label != 1:
                continue
            for w in [w1, w2]:
                if w in concept2id:
                    continue
                concept2id[w] = len(concept2id)
                for c in w:
                    char2id[c] = char2id.get(c, len(char2id))
    print('共有%d个字符' % len(char2id))
    print('共有%d个概念' % len(concept2id))

    encoder = gzh.bert_encoder('mt_bert')

    id2concept = {i: concept2id[i] for i in concept2id}
    id2char = {i: char2id[i] for i in char2id}

    f = open(os.path.join(dataset_path, 'meituan.content'), 'w', encoding='utf-8')
    print('开始计算content')
    time.sleep(0.1)
    batch = []
    ws = []
    for w in tqdm.tqdm(concept2id):
        ws.append(w)
        batch.append(w)
        if len(batch) == 32:
            outputs = encoder(batch)
            for output, w in zip(outputs, ws):
                f.write(str(concept2id[w]) + '\t')
                output = [str(i) for i in output]
                f.write('\t'.join(output) + '\t')
                f.write(w + '\n')
            batch = []
            ws = []
    f.close()

    f = open(os.path.join(dataset_path, 'meituan.graph'), 'w', encoding='utf-8')
    for dataset in [trainData, valData, testData]:
        for w1, w2, label, _ in dataset:
            if label != 1:
                continue
            f.write(str(concept2id[w1]) + '\t' + str(concept2id[w2]) + '\n')
    f.close()
    # 改造结束
    # --------------------------------------------------------------------------- #

    gzh.toJson(id2char, os.path.join(dataset_path, 'id2char.json'))
    gzh.toJson(char2id, os.path.join(dataset_path, 'char2id.json'))
    gzh.toJson(concept2id, os.path.join(dataset_path, 'concept2id.json'))
    gzh.toJson(id2concept, os.path.join(dataset_path, 'id2concept.json'))


if __name__ == '__main__':
    data_path = '../mtdata'
    dataset_path = '../dataset'
    getdataset(data_path, dataset_path)
