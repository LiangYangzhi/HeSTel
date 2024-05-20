import torch
import torch.nn.functional as F


def infoNCE_loss(q, k, temperature=0.1):
    similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分
    logits = similarity_scores / temperature
    N = q.size(0)
    labels = torch.arange(N).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def dis_loss(sim, penalty=0.1):
    diag = torch.diag(sim)
    non_diag = sim - torch.diagflat(diag)
    max_non_diag = non_diag.max(dim=1)[0]
    losses = F.relu(penalty - diag + max_non_diag.unsqueeze(1))
    loss = losses.mean()
    return loss


def diag_loss(sim):
    sim = (sim + 1) / 2
    diag = torch.diag(sim).mean()
    loss = F.tanh(1 - diag).mean()
    return loss


def cosine_loss(sim):
    sim = (sim + 1) / 2
    diag = torch.diag(sim)
    non_diag = sim - torch.diagflat(diag)
    non_diag = non_diag.sum(dim=1)
    diag_loss = F.relu(1 - diag).mean()
    non_diag_loss = F.relu(non_diag).mean()
    loss = diag_loss + non_diag_loss
    return loss


def top_loss(sim, penalty=1):
    row_ranks = sim.argsort(dim=1, descending=True)[:, 0]
    row_ranks += 1
    losses = torch.log(row_ranks ** penalty)
    loss = losses.to(torch.float).mean()
    return loss
