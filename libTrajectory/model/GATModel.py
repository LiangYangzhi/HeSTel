from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=in_dim, out_channels=out_dim,
                             edge_dim=1, add_self_loops=True, heads=heads)
        self.conv2 = GATConv(in_channels=out_dim * heads, out_channels=out_dim,
                             edge_dim=1, add_self_loops=True, heads=heads)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return x
