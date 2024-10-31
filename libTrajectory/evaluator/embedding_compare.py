import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
from libTrajectory.model.hst import HST
from libTrajectory.preprocessing.STEL.graphDataset import GraphLoader
from libTrajectory.preprocessing.STEL.batch_collate import infer_coll_tid
import matplotlib.pyplot as plt


class EmbeddingCompare(object):
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']

    def run(self):
        self.get_tid()
        self.get_init_embedding()
        self.get_model()
        self.get_train_embedding()
        self.dim_reduction()
        return self.init_embedding1, self.init_embedding2, self.train_embedding1, self.train_embedding2

    def get_tid(self):
        tid = pd.read_csv(f"{self.config['path']}{self.config['test_file']}", usecols=['tid'], dtype={'tid': str})
        self.tid = tid.tid.unique().tolist()

    def get_init_embedding(self):
        embedding1 = []
        embedding2 = []
        for t in self.tid:
            graph1 = np.load(f"{self.config['path']}graph1/{t}.npz")
            graph2 = np.load(f"{self.config['path']}graph2/{t}.npz")
            embedding1.append(graph1['node'][0])
            embedding2.append(graph2['node'][0])
        self.init_embedding1 = np.array(embedding1)
        self.init_embedding2 = np.array(embedding2)

    def get_model(self):
        state_dict1 = torch.load(f"{self.config['model_path']}", map_location=self.device)
        self.net = HST(self.config["in_dim"], self.config["out_dim"], self.config["head"]).to(self.device)
        self.net.load_state_dict(state_dict1)
        self.net.eval()

    def get_train_embedding(self):
        graph_data = GraphLoader(self.config['path'], self.tid, train=False)
        data_loader = DataLoader(graph_data, batch_size=16, num_workers=1, collate_fn=infer_coll_tid,
                                 persistent_workers=True)
        embedding1 = []
        embedding2 = []
        for node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, tids in data_loader:  # 每个批次循环
            x = self.net(node, edge, edge_attr, global_spatial, global_temporal)
            vec1 = x[tid1].to(device='cpu')
            vec2 = x[tid2].to(device='cpu')
            for i in range(len(tids)):
                embedding1.append(vec1[i].detach().numpy())
                embedding2.append(vec2[i].detach().numpy())
        self.train_embedding1 = np.array(embedding1)
        self.train_embedding2 = np.array(embedding2)

    def dim_reduction(self):
        # Dimensionality reduction
        svd = TruncatedSVD(n_components=2, algorithm='arpack')
        self.init_embedding1 = svd.fit_transform(self.init_embedding1)
        self.init_embedding2 = svd.fit_transform(self.init_embedding2)
        self.train_embedding1 = svd.fit_transform(self.train_embedding1)
        self.train_embedding2 = svd.fit_transform(self.train_embedding2)
