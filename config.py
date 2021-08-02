'''
@author: Zhouhong GU
@date: 2021/07/29
@target:
'''

# GCN/BERT/GCN+BERT
# use_model = 'GCN'
use_model = 'BERT'
# use_model = 'BERT+GCN'

# position embedding
use_pos_emb = False
pos_emb_size = 128

# Contrastive Learning
use_cl = False

# 预训练模型
bert_pre = 'mt_bert'
# bert_pre = 'chinese-base-bert'

# template
template = lambda x: '%s[SEP]%s' % (x[0], x[1])
# template = lambda x: '%s是一种%s。' % (x[1], x[0])

# deep model 超参数
lr = 1e-3
train_batch_size = 64
test_batch_size = 256
epochs = 10

# GCn 超参数
GCN_layer = 2  # GCN层数
nfeat = 768  # 特征层
nhid = 768  # 隐藏层
nclass = 1  # 分类结果
dropout = 0.4  # dropout
use_cuda = True  # 使用cuda
fastmode = True  # 需不需要验证
test_split = 0.2  # 测试集划分
