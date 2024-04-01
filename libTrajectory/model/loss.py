import torch
import torch.nn.functional as F


def sim_loss(sim, penalty=0.1):
    diag = torch.diag(sim)
    non_diag = sim - torch.diagflat(diag)
    max_non_diag = non_diag.max(dim=1)[0]
    losses = F.relu(penalty - diag + max_non_diag.unsqueeze(1))
    loss = losses.mean()
    return loss


def top_loss(sim, penalty=1):
    row_ranks = sim.argsort(dim=1, descending=True)[:, 0]
    row_ranks += 1
    losses = torch.log(row_ranks ** penalty)
    loss = losses.to(torch.float).mean()
    return loss
