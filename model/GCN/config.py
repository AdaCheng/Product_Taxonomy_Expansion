'''
@author: Zhouhong Gu
@date: 2021/07/26
@target: 各种参数
'''

GCN_layer = 4  # GCN层数
nfeat = 256 # 特征层
nhid = 768  # 隐藏层
nclass = 2  # 分类结果
dropout = 0.4  # dropout
epochs = 200  # epoch
lr = 1e-5  # learning rate
use_cuda = True  # 使用cuda
weight_decay = 5e-4  # adam的参数
fastmode = True  # 需不需要验证
test_split = 0.2  # 测试集划分
