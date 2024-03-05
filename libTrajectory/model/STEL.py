import torch_geometric.nn as pyg_nn
import torch


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_dim, out_channels=out_dim, device=device)
        self.conv2 = pyg_nn.GCNConv(in_channels=out_dim, out_channels=out_dim, device=device)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)

        return x
