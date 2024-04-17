import torch
from libTrajectory.model.transformer_conv import TransformerConv


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(
            in_channels=in_dim, out_channels=out_dim, edge_dim=1, add_self_loops=True, heads=heads, dropout=0.2)
        self.conv2 = TransformerConv(
            in_channels=out_dim*heads, out_channels=out_dim, edge_dim=1, add_self_loops=True, heads=heads, dropout=0.2)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return x


if __name__ == "__main__":
    node1 = [[0, 1, 1, 0], [0, 1, 1, 1]]
    edge1 = [[0, 1], [1, 1]]
    edge_attr1 = [5, -60]

    node2 = [[0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]]
    edge2 = [[0, 1], [1, 2]]
    edge_attr2 = [3, 18]

    node = node1 + node2
    edge = edge1
    for i in edge2[0]:
        edge[0].append(i + len(node1))
    for i in edge2[1]:
        edge[1].append(i + len(node1))
    edge_attr = edge_attr1 + edge_attr2

    node = torch.tensor(node, dtype=torch.float32)
    edge = torch.tensor(edge, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    net = GraphTransformer(len(node[0]), len(node[0]), 2)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    x = net(node, edge, edge_attr)
    print(f"net: {x}")
