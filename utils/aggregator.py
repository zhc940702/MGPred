import torch
import heapq
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.l0dense import L0Dense
from torch.autograd import Variable
from utils.attention import attention



class aggregator(nn.Module):
    """
    aggregator: for aggregating feature of neighbors
    """
    def __init__(self, drug_feature, side_feature, adj, embed_dim, weight_decay = 0.0005, droprate = 0.5,
                    cuda = "cpu", is_drug_part = True):
        super(aggregator, self).__init__()

        self.drugfeature = drug_feature
        self.sidefeature = side_feature
        self.adj = adj
        self.embed_dim = embed_dim
        self.weight_decay = weight_decay
        self.droprate = droprate
        self.device = cuda
        self.is_drug = is_drug_part

        self.u_layer = L0Dense(self.drugfeature.embedding_dim, self.embed_dim,
                                weight_decay = self.weight_decay, droprate = self.droprate)
        self.i_layer = L0Dense(self.sidefeature.embedding_dim, self.embed_dim,
                                weight_decay = self.weight_decay, droprate = self.droprate)
        self.att = attention(self.embed_dim, self.droprate, cuda = self.device)
    def forward(self, nodes):
        if self.is_drug:
            embed_matrix = torch.empty(self.drugfeature.num_embeddings, self.embed_dim,
                                        dtype = torch.float).to(self.device)
            nodes_fea = self.u_layer(self.drugfeature.weight[nodes]).to(self.device)
            threshold = 24
        else:
            embed_matrix = torch.empty(self.sidefeature.num_embeddings, self.embed_dim,
                                        dtype = torch.float).to(self.device)
            nodes_fea = self.i_layer(self.sidefeature.weight[nodes]).to(self.device)
            threshold = 24

        length = list(nodes.size())
        for i in range(length[0]):
            index = nodes[[i]].cpu().numpy()
            if self.training:
                interactions = self.adj[index[0]]
            else:
                if index[0] in self.adj.keys():
                    interactions = self.adj[index[0]]
                else:
                    if self.is_drug:
                        node_feature = self.u_layer(self.drugfeature.weight[torch.LongTensor(index)]).to(self.device)
                    else:
                        node_feature = self.i_layer(self.sidefeature.weight[torch.LongTensor(index)]).to(self.device)

                    embed_matrix[index] = node_feature
                    continue

            n = len(interactions)
            if n < threshold:
                interactions = [item[0] for item in interactions]
                times = n
            else:
                interactions = a_res(interactions, threshold)
                times = threshold
            if self.is_drug:
                neighs_feature = self.i_layer(self.sidefeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.u_layer(self.drugfeature.weight[torch.LongTensor(index)]).to(self.device)
            else:
                neighs_feature = self.u_layer(self.drugfeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.i_layer(self.sidefeature.weight[torch.LongTensor(index)]).to(self.device)
            att_w = self.att(neighs_feature, node_feature, times)
            embedding = torch.mm(neighs_feature.t(), att_w)
            embed_matrix[index] = embedding.t()
        return nodes_fea, embed_matrix

def a_res(samples, m):
    """
    samples: [(entity, weight), ...]
    m: number of selected entities
    returns: [(entity, weight), ...]
    """
    heap = []
    for sample in samples:
        w = sample[1]
        u = random.uniform(0, 1)
        k = u ** (1/w)

        if len(heap) < m:
            heapq.heappush(heap, (k, sample))
        elif k > heap[0][0]:
            heapq.heappush(heap, (k, sample))

            if len(heap) > m:
                heapq.heappop(heap)
    return [item[1][0] for item in heap]
