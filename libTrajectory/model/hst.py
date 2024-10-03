import logging

import torch
from libTrajectory.model.hst_conv import HSTConv
# from torch_geometric.nn import TransformerConv
import torch.nn.functional as F


class HST(torch.nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(HST, self).__init__()
        add_self_loops = True
        logging.info(f"add_self_loops={add_self_loops}")
        self.conv1 = HSTConv(
            in_channels=in_dim, out_channels=out_dim, edge_dim=1,
            add_self_loops=add_self_loops, heads=heads, dropout=0.2)
        self.conv2 = HSTConv(
            in_channels=out_dim*heads, out_channels=out_dim, edge_dim=1,
            add_self_loops=add_self_loops, heads=heads, dropout=0.2)

    def forward(self, x, edge_index, edge_weight=None, global_spatial=None, global_temporal=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight,
                       global_spatial=global_spatial, global_temporal=global_temporal)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight,
                       global_spatial=global_spatial, global_temporal=global_temporal)
        # x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight,
        #                global_spatial=global_spatial, global_temporal=global_temporal)
        # x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight,
        #                global_spatial=global_spatial, global_temporal=global_temporal)
        # x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight,
        #                global_spatial=global_spatial, global_temporal=global_temporal)
        # x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight,
        #                global_spatial=global_spatial, global_temporal=global_temporal)
        return x
