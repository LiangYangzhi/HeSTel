from random import sample
import torch


def twoTower_train(batch):
    # data[key] = (node, edge_ind, edge_attr) or None
    tid1, tid2, ps1, ps2, ns1, ns2 = [], [], [], [], [], []
    # g1
    node1 = []
    edge1 = [[], []]
    edge_attr1 = []
    # g2
    node2 = []
    edge2 = [[], []]
    edge_attr2 = []

    for ind, dic in enumerate(batch):
        add_len1 = len(node1)
        node1 = node1 + dic['g1'][0]
        e_ind0, e_ind1 = dic['g1'][1]
        edge1[0] += [i + add_len1 for i in e_ind0]
        edge1[1] += [i + add_len1 for i in e_ind1]
        edge_attr1.extend(dic['g1'][2])
        tid1.append(add_len1)

        add_len2 = len(node2)
        node2 = node2 + dic['g2'][0]
        e_ind0, e_ind1 = dic['g2'][1]
        edge2[0] += [i + add_len2 for i in e_ind0]
        edge2[1] += [i + add_len2 for i in e_ind1]
        edge_attr2.extend(dic['g2'][2])
        tid2.append(add_len2)

        if dic['ps1'] is not None:
            add_len1 = len(node1)
            node1 = node1 + dic['ps1'][0]
            e_ind0, e_ind1 = dic['ps1'][1]
            edge1[0] += [i + add_len1 for i in e_ind0]
            edge1[1] += [i + add_len1 for i in e_ind1]
            edge_attr1.extend(dic['ps1'][2])
        ps1.append(add_len1)

        if dic['ps2'] is not None:
            add_len2 = len(node2)
            node2 = node2 + dic['ps2'][0]
            e_ind0, e_ind1 = dic['ps2'][1]
            edge2[0] += [i + add_len2 for i in e_ind0]
            edge2[1] += [i + add_len2 for i in e_ind1]
            edge_attr2.extend(dic['ps2'][2])
        ps2.append(add_len2)

        if dic['ns1'] is not None:
            add_len1 = len(node1)
            node1 = node1 + dic['ns1'][0]
            e_ind0, e_ind1 = dic['ns1'][1]
            edge1[0] += [i + add_len1 for i in e_ind0]
            edge1[1] += [i + add_len1 for i in e_ind1]
            edge_attr1.extend(dic['ns1'][2])
            ns1.append(add_len1)
        else:
            ns1.append(None)  # 后续从tid1随机选择

        if dic['ns2'] is not None:
            add_len2 = len(node2)
            node2 = node2 + dic['ns2'][0]
            e_ind0, e_ind1 = dic['ns2'][1]
            edge2[0] += [i + add_len2 for i in e_ind0]
            edge2[1] += [i + add_len2 for i in e_ind1]
            edge_attr2.extend(dic['ns2'][2])
            ns2.append(add_len2)
        else:
            ns2.append(None)  # 后续从tid2随机选择

    # 随机选择tid
    # if None in ns1:
    #     for k, v in enumerate(ns1):
    #         if not v:  # v = None
    #             while True:
    #                 random_ind = sample(tid1, 1)[0]
    #                 if random_ind != tid1[k]:
    #                     ns1[k] = random_ind
    #                     break
    # if None in ns2:
    #     for k, v in enumerate(ns2):
    #         if not v:  # v = None
    #             while True:
    #                 random_ind = sample(tid2, 1)[0]
    #                 if random_ind != tid2[k]:
    #                     ns2[k] = random_ind
    #                     break

    node1 = torch.tensor(node1, dtype=torch.float32)
    edge1 = torch.tensor(edge1, dtype=torch.long)
    edge_attr1 = torch.tensor(edge_attr1, dtype=torch.float32)
    node2 = torch.tensor(node2, dtype=torch.float32)
    edge2 = torch.tensor(edge2, dtype=torch.long)
    edge_attr2 = torch.tensor(edge_attr2, dtype=torch.float32)
    return node1, edge1, edge_attr1, node2, edge2, edge_attr2, tid1, tid2, ps1, ps2, ns1, ns2


def twoTower_infer(batch):
    # data[key] = (node, edge_ind, edge_attr) or None
    tid1, tid2 = [], []
    # g1
    node1 = []
    edge1 = [[], []]
    edge_attr1 = []
    # g2
    node2 = []
    edge2 = [[], []]
    edge_attr2 = []

    for ind, dic in enumerate(batch):
        add_len1 = len(node1)
        node1 = node1 + dic['g1'][0]
        e_ind0, e_ind1 = dic['g1'][1]
        edge1[0] += [i + add_len1 for i in e_ind0]
        edge1[1] += [i + add_len1 for i in e_ind1]
        edge_attr1.extend(dic['g1'][2])
        tid1.append(add_len1)

        add_len2 = len(node2)
        node2 = node2 + dic['g2'][0]
        e_ind0, e_ind1 = dic['g2'][1]
        edge2[0] += [i + add_len2 for i in e_ind0]
        edge2[1] += [i + add_len2 for i in e_ind1]
        edge_attr2.extend(dic['g2'][2])
        tid2.append(add_len2)

    node1 = torch.tensor(node1, dtype=torch.float32)
    edge1 = torch.tensor(edge1, dtype=torch.long)
    edge_attr1 = torch.tensor(edge_attr1, dtype=torch.float32)
    node2 = torch.tensor(node2, dtype=torch.float32)
    edge2 = torch.tensor(edge2, dtype=torch.long)
    edge_attr2 = torch.tensor(edge_attr2, dtype=torch.float32)
    return node1, edge1, edge_attr1, node2, edge2, edge_attr2, tid1, tid2


def train_coll(batch):
    tid1, tid2, ps1, ps2, ns1, ns2 = [], [], [], [], [], []
    node = []
    edge = [[], []]
    edge_attr = []
    global_spatial = []
    global_temporal = []

    for dic in batch:    # dic[key] = (node, edge_ind, edge_attr) or None
        add_len = len(node)
        node = node + dic['g1'][0]
        edge0, edge1 = dic['g1'][1]
        edge[0] += [i + add_len for i in edge0]
        edge[1] += [i + add_len for i in edge1]
        edge_attr.extend(dic['g1'][2])
        global_spatial = global_spatial + dic['g1'][3]
        global_temporal = global_temporal + dic['g1'][4]
        tid1.append(add_len)
        # if dic['ps1'] is not None:
        #     add_len = len(node)
        #     node = node + dic['ps1'][0]
        #     edge0, edge1 = dic['ps1'][1]
        #     edge[0] += [i + add_len for i in edge0]
        #     edge[1] += [i + add_len for i in edge1]
        #     edge_attr.extend(dic['ps1'][2])
        # ps1.append(add_len)

        add_len = len(node)
        node = node + dic['g2'][0]
        edge0, edge1 = dic['g2'][1]
        edge[0] += [i + add_len for i in edge0]
        edge[1] += [i + add_len for i in edge1]
        edge_attr.extend(dic['g2'][2])
        global_spatial = global_spatial + dic['g2'][3]
        global_temporal = global_temporal + dic['g2'][4]
        tid2.append(add_len)
        # if dic['ps2'] is not None:
        #     add_len = len(node)
        #     node = node + dic['ps2'][0]
        #     edge0, edge1 = dic['ps2'][1]
        #     edge[0] += [i + add_len for i in edge0]
        #     edge[1] += [i + add_len for i in edge1]
        #     edge_attr.extend(dic['ps2'][2])
        # ps2.append(add_len)

        # if dic['ns1'] is not None:
        #     lis = []
        #     for ns in dic['ns1'].values():
        #         add_len = len(node)
        #         node = node + ns[0]
        #         edge0, edge1 = ns[1]
        #         edge[0] += [i + add_len for i in edge0]
        #         edge[1] += [i + add_len for i in edge1]
        #         edge_attr.extend(ns[2])
        #         lis.append(add_len)
        #     ns1.append(lis)
        # else:
        #     ns1.append(None)
        #
        # if dic['ns2'] is not None:
        #     lis = []
        #     for ns in dic['ns2'].values():
        #         add_len = len(node)
        #         node = node + ns[0]
        #         edge0, edge1 = ns[1]
        #         edge[0] += [i + add_len for i in edge0]
        #         edge[1] += [i + add_len for i in edge1]
        #         edge_attr.extend(ns[2])
        #         lis.append(add_len)
        #     ns2.append(lis)
        # else:
        #     ns2.append(None)

    # # 随机选择tid
    # for k, v in enumerate(ns1):
    #     add_num = len(batch) if v is None else len(batch) - len(v)
    #     if add_num == 0:
    #         continue
    #     if add_num == len(batch):
    #         add_ind = tid1.copy()
    #         add_ind.remove(tid1[k])
    #         add_ind.append(add_ind[-1])
    #     else:
    #         add_ind = sample(tid1, add_num + 1)
    #         if tid1[k] in add_ind:
    #             add_ind.remove(tid1[k])
    #         else:
    #             add_ind = add_ind[: -1]
    #     ns_list = ns1[k] + add_ind
    #     ns1[k] = ns_list
    #
    # for k, v in enumerate(ns2):
    #     add_num = len(batch) if v is None else len(batch) - len(v)
    #     if add_num == 0:
    #         continue
    #     if add_num == len(batch):
    #         add_ind = tid2.copy()
    #         add_ind.remove(tid2[k])
    #         add_ind.append(add_ind[-2])
    #     else:
    #         add_ind = sample(tid2, add_num + 2)
    #         if tid2[k] in add_ind:
    #             add_ind.remove(tid2[k])
    #         else:
    #             add_ind = add_ind[: -2]
    #     ns_list = ns2[k] + add_ind
    #     ns2[k] = ns_list
    #
    # for i in range(len(batch)):
    #     ns1[i][i] = tid1[i]
    #     ns1[i] = ns1[i][: len(batch)]
    #
    # for i in range(len(batch)):
    #     ns2[i][i] = tid2[i]
    #     ns2[i] = ns2[i][: len(batch)]

    node = torch.tensor(node, dtype=torch.float32)
    edge = torch.tensor(edge, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    global_spatial = torch.tensor(global_spatial, dtype=torch.float32)
    global_temporal = torch.tensor(global_temporal, dtype=torch.float32)
    return node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, ps1, ps2, ns1, ns2


def infer_coll(batch):
    tid1, tid2 = [], []
    node = []
    edge = [[], []]
    edge_attr = []
    global_spatial = []
    global_temporal = []

    for ind, dic in enumerate(batch):  # dic[key] = (node, edge_ind, edge_attr) or None
        add_len = len(node)
        node = node + dic['g1'][0]
        edge0, edge1 = dic['g1'][1]
        edge[0] += [i + add_len for i in edge0]
        edge[1] += [i + add_len for i in edge1]
        edge_attr.extend(dic['g1'][2])
        global_spatial = global_spatial + dic['g1'][3]
        global_temporal = global_temporal + dic['g1'][4]
        tid1.append(add_len)

        add_len = len(node)
        node = node + dic['g2'][0]
        edge0, edge1 = dic['g2'][1]
        edge[0] += [i + add_len for i in edge0]
        edge[1] += [i + add_len for i in edge1]
        edge_attr.extend(dic['g2'][2])
        global_spatial = global_spatial + dic['g2'][3]
        global_temporal = global_temporal + dic['g2'][4]
        tid2.append(add_len)

    node = torch.tensor(node, dtype=torch.float32)
    edge = torch.tensor(edge, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    global_spatial = torch.tensor(global_spatial, dtype=torch.float32)
    global_temporal = torch.tensor(global_temporal, dtype=torch.float32)
    return node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2
