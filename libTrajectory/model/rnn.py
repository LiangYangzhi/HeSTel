import torch
# from torch_geometric.nn import TransformerConv


def rnn(config):
    return torch.nn.RNN(**config)  #
