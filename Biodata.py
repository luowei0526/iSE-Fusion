import numpy as np
import torch
from torch_geometric.data import Data
import encode_seq
import sys


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset():
    def __init__(self, pnode_feature, fnode_feature, edge):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.edge = edge

    def process(self):
        data_list = []
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)

            data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index)

            data_list.append(data)

        return data_list


class Biodata:
    def __init__(self, seq, K=3, d=3, seqtype="DNA"):
        self.seq = seq
        self.K = K
        self.d = d
        self.seqtype = seqtype

        self.edge = []
        for i in range(4 ** (K * 2)):
            a = i // 4 ** K
            b = i % 4 ** K
            self.edge.append([a, i])
            self.edge.append([b, i])
        self.edge = np.array(self.edge).T


    def encode(self):
        # 编码单个序列
        feature = encode_seq.matrix_encoding(self.seq, K=self.K, d=self.d, seqtype=self.seqtype) # shape: (12288,)
        feature = feature.reshape(self.d, 4 ** self.K, 4 ** self.K) # 恢复为 [d, 4^K, 4^K]
        feature = np.expand_dims(feature, axis=0)  # shape: [1, d, 4^K, 4^K] 

        # patch-node feature: [1, d, 4^(K*2)] -> [1, 4^(K*2), d]
        self.pnode_feature = feature.reshape(-1, self.d, 4 ** (self.K * 2))
        self.pnode_feature = np.moveaxis(self.pnode_feature, 1, 2)

        # f-node feature: 从 d=0 的层计算 4^K 个 f-node，每个是 1 维
        zero_layer = feature[:, 0, :, :]
        self.fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4 ** self.K, 1)

        # 只处理一个图
        graph = GraphDataset(self.pnode_feature, self.fnode_feature, self.edge)
        data = graph.process()[0]  # 返回列表中的第一个

        return data
