import torch_geometric.nn as pyg_nn
import torch


class Graph(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Graph, self).__init__()
        self.conv1 = pyg_nn.GraphConv(in_dim, out_dim, add_self_loops=True)
        self.conv2 = pyg_nn.GraphConv(out_dim, out_dim, add_self_loops=True)

    def forward(self, x, edge_index, edge_weight=None, global_spatial=None, global_temporal=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(GAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels=in_dim, out_channels=out_dim,
                                    edge_dim=1, add_self_loops=True, heads=heads, dropout=0.2)
        self.conv2 = pyg_nn.GATConv(in_channels=out_dim * heads, out_channels=out_dim,
                                    edge_dim=1, add_self_loops=True, heads=heads, dropout=0.2)

    def forward(self, x, edge_index, edge_weight=None, global_spatial=None, global_temporal=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_dim, out_channels=out_dim, add_self_loops=True)
        self.conv2 = pyg_nn.GCNConv(in_channels=out_dim, out_channels=out_dim, add_self_loops=True)

    def forward(self, x, edge_index, edge_weight=None, global_spatial=None, global_temporal=None):
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x
