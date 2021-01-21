import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class attention(nn.Module):
    def __init__(self, embedding_dim, droprate, cuda = "cpu"):
        super(attention, self).__init__()

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax()

    # a = np.array([[1,2,3]])
    # a = a.repeat(5, 1)
    # [[1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]]
    # 论文中公式（4）中atm在哪？ 就是第一层隐藏层中的待训练的w把？
    def forward(self, feature1, feature2, n_neighs):
        # feature2 = feature2.detach().numpy()
        # print(feature1.detach().numpy())
        # print(feature2.detach().numpy())
        # print(n_neighs)
        # 先传该节点所有邻居的特征，然后传该节点的特征
        feature2_reps = feature2.repeat(n_neighs, 1)

        x = torch.cat((feature1, feature2_reps), 1)
        x = F.relu(self.att1(x).to(self.device), inplace=True)
        x = F.dropout(x, training=self.training, p=self.droprate)
        # print(x.detach().numpy())
        x = self.att2(x).to(self.device)

        att = F.softmax(x, dim=0)
        # xxxx = att.detach().numpy() # 返回一个样本的所有邻居的各自权重的值，然后把它标准化
        return att