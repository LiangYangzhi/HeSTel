from torch_geometric.nn import GraphConv
import torch
import torch.nn.functional as F


class Graph(torch.nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(Graph, self).__init__()
        self.conv1 = GraphConv(in_dim, out_dim, add_self_loops=True, aggr="mean")
        self.conv2 = GraphConv(out_dim, out_dim*heads, add_self_loops=True, aggr="mean")

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


