import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F


class TowerGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TowerGCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
        self.conv2 = pyg_nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
        self.pooling = torch.nn.AvgPool1d(kernel_size=out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # [num_nodes, num_node_features] --> [1, num_nodes, num_node_features]
        x = x.unsqueeze(0)
        # [1, num_nodes, num_node_features] --> [1, num_node_features, num_nodes] --> [1, 1, num_nodes] --> [num_nodes]
        x = self.pooling(x.permute(0, 2, 1)).squeeze()  # Pool over the feature dimension
        return x


class DualTowerGCN(torch.nn.Module):
    def __init__(self, in_dim1, out_dim1, in_dim2, out_dim2):
        super(DualTowerGCN, self).__init__()
        self.tower1 = TowerGCN(in_dim1, out_dim1)
        self.tower2 = TowerGCN(in_dim2, out_dim2)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        # x1, x2: [num_nodes, num_node_features]
        # edge_index1, edge_index2: [2, num_edges]
        # edge_weight1, edge_weight2: [num_edges]
        v1 = self.tower1(x1, edge_index1, edge_weight1)
        v2 = self.tower2(x2, edge_index2, edge_weight2)

        # Concatenate tower outputs
        v = torch.cat((v1, v2), dim=1)

        # Final classification
        out = self.fc(v)
        return torch.sigmoid(out)
