import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut


class TowerGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TowerGCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
        self.conv2 = pyg_nn.GCNConv(in_channels=out_dim, out_channels=out_dim)
        self.fc = torch.nn.Linear(64, 64)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # x = x.type(torch.float32)
        # edge_index = edge_index.long()
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # [num_nodes, num_node_features] --> [1, num_nodes, num_node_features]
        # x = x.unsqueeze(0)
        # [1, num_nodes, num_node_features] --> [1, num_node_features, num_nodes] --> [1, 1, num_nodes] --> [num_nodes]
        # x = self.pooling(x.permute(0, 2, 1)).squeeze()  # Pool over the feature dimension
        cluster = pyg_nn.graclus(edge_index, edge_weight, x.size(0))
        x, batch = pyg_nn.pool.max_pool_x(cluster, x, batch)

        entity = []  # 节点平均值作为实体表示
        for i in torch.unique(batch):
            ind = torch.where(batch == i)
            if ind[0].shape[0] == 1:
                node = x[ind[0][0]]
            else:
                node = x[ind[0][0]: ind[0][1], :]
                node = node.mean(dim=0, keepdim=True)[0]
            entity.append(node)
        return torch.stack(entity)
