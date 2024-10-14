from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F


class GCN1(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(in_dim, out_dim, improved=True, add_self_loops=True)
        self.conv2 = GCNConv(out_dim, out_dim, improved=True, add_self_loops=True)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
            edge_weight = F.relu(edge_weight)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class TwinTowerGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TwinTowerGCN, self).__init__()
        self.gcn1 = GCN1(in_dim, out_dim)
        self.gcn2 = GCN1(in_dim, out_dim)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        g1 = self.gcn1(x=x1, edge_index=edge_index1, edge_weight=edge_weight1)
        g2 = self.gcn1(x=x2, edge_index=edge_index2, edge_weight=edge_weight2)
        return g1, g2

