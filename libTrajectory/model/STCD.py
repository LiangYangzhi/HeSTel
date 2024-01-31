import torch_geometric.nn as pyg_nn
import torch


class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads):
        super(GraphTransformer, self).__init__()
        self.conv1 = pyg_nn.TransformerConv(
            in_channels=input_dim, out_channels=output_dim, edge_dim=1, heads=heads, dropout=0.1, beta=True)
        self.conv2 = pyg_nn.TransformerConv(
            in_channels=output_dim*heads, out_channels=output_dim, edge_dim=1, heads=heads, dropout=0.1, beta=True)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.conv2(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.conv2(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x = x+x1 #res connect
        return x


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_num}')
    print(f'Total number of trainable parameters: {trainable_num}')
    return {'Total': total_num, 'Trainable': trainable_num}
