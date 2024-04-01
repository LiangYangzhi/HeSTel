import torch_geometric.nn as pyg_nn
import torch


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(GraphTransformer, self).__init__()
        self.conv1 = pyg_nn.TransformerConv(
            in_channels=in_dim, out_channels=out_dim, edge_dim=1, heads=heads, dropout=0.1, beta=True)
        self.conv2 = pyg_nn.TransformerConv(
            in_channels=out_dim*heads, out_channels=out_dim, edge_dim=1, heads=heads, dropout=0.1, beta=True)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return x
