import torch


def rnn(config):
    return torch.nn.RNN(**config)  #
