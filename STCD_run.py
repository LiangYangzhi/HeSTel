import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F


class EntityTower(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EntityTower, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=input_dim, out_channels=output_dim)
        self.conv2 = pyg_nn.GCNConv(in_channels=output_dim, out_channels=output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        return x


class DualTowerGCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DualTowerGCN, self).__init__()
        self.tower1 = EntityTower(input_dim, output_dim)
        self.tower2 = EntityTower(input_dim, output_dim)
        self.fc = torch.nn.Linear(2 * output_dim, 1)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        # x1, x2: [num_nodes, num_node_features]
        # edge_index1, edge_index2: [2, num_edges]
        # edge_weight1, edge_weight2: [num_edges]
        h1 = self.tower1(x1, edge_index1, edge_weight1)
        h2 = self.tower2(x2, edge_index2, edge_weight2)

        # Aggregate entity representations (e.g., average pooling)
        h1 = torch.mean(h1, dim=0)
        h2 = torch.mean(h2, dim=0)

        # Concatenate tower outputs
        h = torch.cat((h1, h2), dim=1)

        # Final classification
        out = self.fc(h)
        return torch.sigmoid(out)


# Usage

"""
在这个例子中，EntityTower类是一个图卷积网络，它将处理每个实体的图数据。
DualTowerGCN类包含两个EntityTower实例，每个实例处理一个实体的图数据。
最后，通过一个全连接层将两个塔的输出合并，并输出一个关系分数，该分数可以通过sigmoid函数转换为概率。
请注意，这个例子假设每个实体都有自己的图数据（节点特征、边索引和可选的边权重）。
在实际应用中，你可能需要根据具体情况调整模型结构和输入数据的处理方式。
"""

model = DualTowerGCN(input_dim=feature_dim, output_dim=hidden_dim)
entity1_representation = model.tower1(entity1_features, entity1_edge_index, entity1_edge_weight)
entity2_representation = model.tower2(entity2_features, entity2_edge_index, entity2_edge_weight)
relation_score = model(entity1_features, entity1_edge_index, entity1_edge_weight, entity2_features, entity2_edge_index,
                       entity2_edge_weight)
