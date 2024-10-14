from torch_geometric.nn import GraphConv
import torch


class Graph1(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Graph1, self).__init__()
        self.conv1 = GraphConv(in_dim, out_dim, add_self_loops=True)
        self.conv2 = GraphConv(out_dim, out_dim, add_self_loops=True)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


