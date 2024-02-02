import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut


class TowerGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TowerGCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
        self.conv2 = pyg_nn.GCNConv(in_channels=out_dim, out_channels=out_dim)

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
        x, clusterid = pyg_nn.pool.max_pool_x(cluster, x, batch)
        return x, clusterid


class DualTowerGCN(torch.nn.Module):
    def __init__(self, in_dim1, out_dim1, in_dim2, out_dim2):
        super(DualTowerGCN, self).__init__()
        self.tower1 = TowerGCN(in_dim1, out_dim1)
        self.tower2 = TowerGCN(in_dim2, out_dim2)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2, batch1=None, batch2=None):
        # x1, x2: [num_nodes, num_node_features]
        # edge_index1, edge_index2: [2, num_edges]
        # edge_weight1, edge_weight2: [num_edges]
        print(x1.shape)
        v1, batch1 = self.tower1(x1, edge_index1, edge_weight1, batch1)
        v2, batch2 = self.tower2(x2, edge_index2, edge_weight2, batch2)

        # Concatenate tower outputs
        print(batch1)
        print(batch2)
        print(v1.shape)
        print(v2.shape)
        v = torch.cat((v1, v2), dim=0)

        # Final classification
        out = self.fc(v)
        return torch.sigmoid(out)
