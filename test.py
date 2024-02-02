# import torch
# # 给定的Tensor
# tensor = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3,
#                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#                        5, 5])
#
# # 获取每个数值的第一个和最后一个索引值
# ind = torch.where(tensor == 1)
# print(ind)
# print(ind[0][0], ind[0][-1])
#
#
# print()
# for i in torch.unique(tensor):
#     print(i)
# print(tensor[1: 2])
#
# e1 = []
# for i in torch.unique(batch1):
#     ind1 = torch.where(batch1)
#     node = v1[ind1[0][0]: ind1[0][-1], :]
#     print(node)
# #
# #
# # import torch
# # import torch.nn as nn
# #
# # # 假设你有一个形状为 [27, 64] 的张量 tensor
# # tensor = torch.randn(27, 64)
# # print(tensor.shape)
# #
# # # 定义一个线性变换层，将输入的维度压缩为 [1, 64]
# # linear_layer = nn.Linear(64, 64)
# #
# # print(tensor)
# #
# # m = tensor.mean(dim=0, keepdim=True)
# # print(m)
# # print(m.shape)
# # # 将张量压缩成 [1, 64]
# # compressed_tensor = linear_layer(tensor)
# #
# # # 打印结果
# # print(compressed_tensor.size())
# #
# #
#
#
# # import torch
# #
# # # 假设lis是一个包含两个张量的列表
# # tensor1 = torch.tensor([1, 2, 3])
# # tensor2 = torch.tensor([4, 5, 6])
# # lis = [tensor1, tensor2]
# #
# # # 使用torch.stack将lis中的张量堆叠成一个新的张量
# # stacked_tensor = torch.stack(lis)
# #
# # print(stacked_tensor)
#


import torch
import torch.nn.functional as F

# 假设有两个tensor v1 和 v2，它们的形状分别为 (m, n) 和 (p, n)
v1 = torch.randn(3, 5)  # 3行5列的tensor
v2 = torch.randn(2, 5)  # 2行5列的tensor

# 直接计算余弦距离
cosine_sim = F.cosine_similarity(v1, v2, dim=1)  # 在第一维度上计算余弦相似度

print(cosine_sim)
