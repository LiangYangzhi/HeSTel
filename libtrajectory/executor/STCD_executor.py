import random
import copy
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import networkx as nx

from libtrajectory.model.STCD_model import GraphTransformer



class Executor(object):
    def __init__(self, check_in, label_edge, placeid_model, spatial_model,
                 node_list, spatial_edge, temporal_edge,
                 batch_size, num_workers, select_label, input_dim, output_dim, lr,
                 log_dir, gpu_id, heads, epoch_num, check_point_interval,
                 weighted_remove, weighted_add, spatial_add_prob, topn_select,
                 embedding_selector, user_embedding_init, aug_num):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.check_in = check_in
        self.label_edge = label_edge
        self.placeid_model = placeid_model
        self.spatial_model = spatial_model
        self.node_list = node_list
        self.spatial_edge = spatial_edge
        self.temporal_edge = temporal_edge
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.select_label = select_label
        self.embedding_selector = embedding_selector
        self.input_dim = input_dim
        if self.embedding_selector == 1:
            self.input_dim = 128
        elif self.embedding_selector == -1:
            self.input_dim = 512

        self.output_dim = output_dim
        self.lr = lr
        self.log_dir = log_dir
        self.gpu_id = gpu_id
        self.heads = heads
        self.epoch_num = epoch_num
        self.check_point_interval = check_point_interval
        self.weighted_remove = weighted_remove
        self.weighted_add = weighted_add
        self.spatial_add_prob = spatial_add_prob
        self.topn_select = topn_select
        self.user_embedding_init = user_embedding_init
        self.aug_num = aug_num

        if torch.cuda.is_available():
            self.use_gpu = True
            print('training on GPU mode')
        else:
            self.use_gpu = False
            print('training on CPU mode')

        self.model = GraphTransformer(self.input_dim, self.output_dim, self.heads)

    def _data_loader(self, train):
        idx_set = IndexDataset(self.check_in, train=train)
        dataloader = torch.utils.data.DataLoader(
            dataset=idx_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        g_set = GraphDataset(self.check_in, self.placeid_model, self.spatial_model, self.spatial_edge,
                             self.temporal_edge, train=train, select_label=self.select_label, topn=self.topn_select)
        return dataloader, g_set

    def train(self):
        train_dataloader, g_set = self._data_loader(train=True)
        opt = torch.optim.Adam(self.model.parameters(), self.lr)
        cost = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        writer = SummaryWriter(log_dir=self.log_dir)
        global_step = 0
        print_n = int(len(train_dataloader) / 10)

        for epoch in range(self.epoch_num):
            t0 = time.time()
            running_loss = 0.0
            print("-" * 10)
            print(f'Epoch {epoch + 1}/{self.epoch_num}')

            for data in tqdm(train_dataloader):
                node_embedding, edge_index, edge_attr, target_user, target_place = g_set[data.tolist()]
                node_embedding = torch.tensor(node_embedding)
                edge_index = torch.tensor(edge_index)
                edge_attr = torch.tensor(edge_attr)
                label = torch.tensor([i for i in range(len(target_user))])
                user_target = []
                for i in range((len(data) * 2)):
                    if i % 2 == 0:
                        user_target.append(i + 1)
                    else:
                        user_target.append(i - 1)
                user_target = torch.tensor(user_target)

                if self.use_gpu:
                    node_embedding = node_embedding.to(device=self.device)
                    edge_index = edge_index.to(device=self.device)
                    edge_attr = edge_attr.to(device=self.device)
                    label = label.to(device=self.device)
                    user_target = user_target.to(device=self.device)

                x = self.model(node_embedding, edge_index, edge_attr)

                target_user_output = x[target_user]
                target_place_output = x[target_place]

                user_output = x[0:(len(data) * 2)]
                sim_user = torch.matmul(user_output, user_output.T)
                sim_user_diag = torch.diag(sim_user)
                sim_user_diag = torch.diag_embed(sim_user_diag)
                sim_user = sim_user - sim_user_diag

                sim_edge = torch.matmul(target_user_output, target_place_output.T)
                loss_mask = cost(sim_edge, label)
                loss_user_aug = cost(sim_user, user_target)
                loss = loss_mask + loss_user_aug

                opt.zero_grad()
                loss.backward()
                opt.step()

                running_loss += loss.data.item()

                writer.add_scalar('BatchLoss/edgeMaskLoss', loss_mask.data.item(), global_step=global_step)
                writer.add_scalar('BatchLoss/userSimLoss', loss_user_aug.data.item(), global_step=global_step)
                writer.add_scalar('BatchLoss/LossSum', loss.data.item(), global_step=global_step)
                global_step = global_step + 1

            if epoch % self.check_point_interval == 0:
                torch.save(self.model.state_dict(), f'{self.log_dir}net_parameter-epoch:{epoch}.pth')
            epoch_loss = running_loss / len(train_dataloader)
            writer.add_scalar('EpochLoss', epoch_loss, global_step=epoch)
            t1 = time.time()
            print(f'Loss is {epoch_loss}, takes {t1 - t0} seconds')
        torch.save(self.model.state_dict(), f'{self.log_dir}net_parameter.pth')

    def infer(self):
        G_real = self._construct_real_network()
        test_dataloader, g_set = self._data_loader(train=False)

        self.model = GraphTransformer(self.input_dim, self.output_dim, self.heads)
        parameter = self.log_dir + 'net_parameter.pth'
        state_dict = torch.load(parameter)
        if self.use_gpu:
            self.model = self.model.to(device=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        user_list = []
        embedding_list = []
        print('network is inferring...')
        for data in tqdm(test_dataloader):
            users = data.list()
            node_embedding, edge_index, edge_attr = g_set[users]
            user_list = user_list + users

            node_embedding = torch.tensor(node_embedding)
            edge_index = torch.tensor(edge_index)
            edge_attr = torch.tensor(edge_attr)
            if self.use_gpu:
                node_embedding = node_embedding.to(device=self.device)
                edge_index = edge_index.to(device=self.device)
                edge_attr = edge_attr.to(device=self.device)

            x = self.model(node_embedding, edge_index, edge_attr)
            number_of_user = len(users)
            embedding = x[0: number_of_user]
            embedding_list.append(copy.deepcopy(embedding.cpu().numpy()))
        embedding = np.vstack(embedding_list)
        return embedding, user_list

    def _construct_real_network(self):
        print("constructing real network...")

        user_set = set(self.check_in['userid'].unique())
        self.label_edge['select'] = self.label_edge.apply(
            lambda row: True if row['userid0'] in user_set and row['userid1'] in user_set else False, axis=1)
        self.label_edge = self.label_edge[self.label_edge['select'] == True]

        G_real = nx.Graph()
        G_real.add_nodes_from(user_set)
        edge_list_construct = [(row['userid0'], row['userid1']) for i, row in self.label_edge.iterrows()]
        G_real.add_edges_from(edge_list_construct)
        return G_real


class IndexDataset(Dataset):
    def __init__(self, data, train=True):
        self.train = train
        self.user_id = list(data['userid'].unique())

    def __len__(self):
        # if self.train:
        #     return len(self.user_id)
        # else:
        #     return 1
        return len(self.user_id)

    def __getitem__(self, index):
        # if self.train:
        #     return self.user_id[index]
        # else:
        #     return self.user_id
        return self.user_id[index]


class GraphDataset(Dataset):
    def __init__(self, data, placeid_model, spatial_model, spatial_edge, temporal_edge, train=False,
                 select_label=64, spatial_add_prob=0.5, add_prob=0.5, min_len=5, max_len=4096,
                 aug_num=20, topn=3, weighted_add=True, weighted_remove=True,
                 embedding_selector=0, user_embedding_init=0):
        self.check_in_group = data.groupby('userid')
        self.placeid_model = placeid_model
        self.spatial_model = spatial_model
        self.train = train
        self.select_label = select_label
        self.topn = topn

        # parameter for data augumentation
        self.spatial_edge = spatial_edge
        self.temporal_edge = temporal_edge
        self.spatial_add_prob = spatial_add_prob
        self.add_prob = add_prob
        self.min_len = min_len
        self.max_len = max_len
        self.aug_num = aug_num
        self.weighted_add = weighted_add
        self.weighted_remove = weighted_remove

        self.embedding_selector = embedding_selector
        self.user_embedding_init = user_embedding_init

    def __len__(self):
        if not self.train:
            return 1
        else:
            return self.check_in_group.ngroups

    def __getitem__(self, index):
        return self.get_sample(index)

    def _random_add(self, seed_p, graph):
        places = Counter(seed_p)
        places_ids = list(places.keys())
        weights = list(places.values()) if self.weighted_add else [1] * len(places_ids)
        places_start = random.choices(population=places_ids, weights=weights, k=self.aug_num)
        places_add = []
        for p in places_start:
            if p in graph:
                weight = graph[p]
                next_p = random.choices(population=list(weight.keys()),
                                        weights=list(weight.values()))[0]
            else:
                next_p = random.choice(list(graph.keys()))
            places_add.append(next_p)
        ans_p = seed_p + places_add
        return ans_p

    def _random_remove(self, seed_p):
        if self.weighted_remove:
            places = Counter(seed_p)
            places_ids = list(places.keys())
            if len(places_ids) > 1:
                weights = np.array(list(places.values()))
                weights = 1 / weights
                p = random.choices(population=places_ids, weights=weights)[0]
                ans_p = [x for x in seed_p if x != p]
            else:
                i = random.randint(0, len(seed_p) -1)
                ans_p = copy.deepcopy(seed_p)
                del ans_p[i]
        else:
            places = set(seed_p)
            if len(places) == 1:
                i = random.randint(0, len(seed_p) - 1)
                ans_p = copy.deepcopy(seed_p)
                del ans_p[i]
            else:
                p = random.choice(places)
                ans_p = [x for x in seed_p if x != p]
        return ans_p

    def _get_aug_sample(self, seed_p):
        if self.min_len > 1 and len(seed_p) <= self.min_len:
            prob = random.random()
            if prob < self.spatial_add_prob:
                return self._random_add(seed_p=seed_p, graph=self.spatial_edge)
            else:
                return self._random_add(seed_p=seed_p, graph=self.temporal_edge)

        elif 1 < self.max_len < len(seed_p):
            return self._random_remove(seed_p)

        else:
            prob = random.random()
            if prob < self.add_prob:
                prob = random.random()
                if prob < self.spatial_add_prob:
                    return self._random_add(seed_p=seed_p, graph=self.spatial_edge)
                else:
                    return self._random_add(seed_p=seed_p, graph=self.temporal_edge)
            else:
                return self._random_remove(seed_p)

    def _get_sample_data(self, spatial_seq, user, topn=3):
        placeid_seq = [str(i) for i in spatial_seq]
        placeid_counter = Counter(placeid_seq)
        top_n = placeid_counter.most_common(topn)
        placeid_embedding = self.placeid_model.wv[placeid_seq]
        spatial_embedding = self.spatial_model.wv[spatial_seq]
        if self.embedding_selector == 1:
            arr = placeid_embedding
        elif self.embedding_selector == -1:
            arr = spatial_embedding
        else:
            arr = np.hstack([placeid_embedding, spatial_embedding])

        topn_place = []
        for p, _ in top_n:
            topn_place.append(p)
        placeid_seq_topn = [i for i in placeid_seq if i in topn_place]
        spatial_seq_topn = [int(i) for i in placeid_seq_topn]

        placeid_embedding_topn = self.placeid_model.wv[placeid_seq_topn]
        spatial_embedding_topn = self.spatial_model.wv[spatial_seq_topn]
        if self.embedding_selector == 1:
            arr_topn = placeid_embedding_topn
        elif self.embedding_selector == -1:
            arr_topn = spatial_embedding_topn
        else:
            arr_topn = np.hstack([placeid_embedding_topn, spatial_embedding_topn])

        if len(placeid_seq) > 0:
            arr = np.mean(arr, axis=0)
            arr_topn = np.mean(arr_topn, axis=0)

        if self.user_embedding_init == 1:
            arr = arr  # global average
        elif self.user_embedding_init == -1:
            arr = arr_topn  # top3 average
        else:
            arr = (arr + arr_topn) / 2

        edges = [{'user': user, 'place': e, 'number': placeid_counter[e]} for e in placeid_counter]
        return arr, placeid_seq, edges

    def get_sample(self, index):
        user_list = index
        placeid_list = []
        sptial_list = []
        node_embedding = []
        edge_df = []
        real_user_list = []

        for i in user_list:
            data = self.check_in_group.get_group(i)
            spatial_seq = list(data['placeid'])
            arr, placeid_seq, edges = self._get_sample_data(spatial_seq, i)
            node_embedding.append(arr)
            placeid_list = placeid_list + placeid_seq
            sptial_list = sptial_list + spatial_seq
            real_user_list.append(i)
            edge_df = edge_df + edges

            if self.train:
                seed_p = self._get_aug_sample(spatial_seq)
                user = str(i) + '-' + 'aug'
                arr, placeid_seq, edges = self._get_sample_data(seed_p, user)
                node_embedding.append(arr)
                placeid_list = placeid_list + placeid_seq
                sptial_list = sptial_list + seed_p
                real_user_list.append(user)
                edge_df = edge_df + edges

        number_of_user = len(real_user_list)
        placeid_list = list(set(placeid_list))
        sptial_list = list(set(sptial_list))
        user_dict = {user: ind for ind, user in enumerate(real_user_list)}
        placeid_dict = {placeid: ind + number_of_user for ind, placeid in enumerate(placeid_list)}
        number_of_place = len(placeid_list)
        node_embedding = np.vstack(node_embedding)

        place_embedding = self.placeid_model.wv[placeid_list]
        spatial_embedding = self.spatial_model.wv[sptial_list]
        if self.embedding_selector == 1:
            placeid_embeddding = place_embedding
        elif self.embedding_selector == -1:
            placeid_embeddding = spatial_embedding
        else:
            placeid_embeddding = np.hstack([place_embedding, spatial_embedding])
        node_embedding = np.vstack([node_embedding, placeid_embeddding])

        edge_df = pd.DataFrame(edge_df)
        edge_df['user'] = edge_df['user'].map(lambda x: user_dict[x])
        edge_df['place'] = edge_df['place'].map(lambda x: placeid_dict[x])

        if self.train:
            select_label = int(self.select_label * len(edge_df))

            edge_df_target = edge_df.sample(n=select_label, replace=False)
            target_user = set(list(edge_df_target['user'].unique()))
            target_place = set(list(edge_df_target['place'].unique()))
            edge_df['select'] = edge_df.apply(
                lambda x: True if x['user'] in target_user and x['place'] in target_place else False, axis=1)
            edge_df_target = edge_df[edge_df['select'] == True]
            edge_df = edge_df[~edge_df.index.isin(edge_df_target.index)]

            edge_index = np.array(edge_df[['user', 'place']]).T
            edge_attr = np.array(edge_df['number'], dtype=np.float32)

            target_user = list(edge_df_target['user'])
            target_place = list(edge_df_target['place'])

            return node_embedding, edge_index, edge_attr, target_user, target_place

        else:
            edge_index = np.array(edge_df[['user', 'place']]).T
            edge_attr = np.array(edge_df['number'], dtype=np.float32)
            return node_embedding, edge_index, edge_attr
