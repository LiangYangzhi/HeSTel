from random import sample
import torch


def train_coll(batch):
    """
    tid: [tid用户索引, tid用户索引, tid用户索引]
    ps: [ps_tid用户索引, ps_tid用户索引， ps用户索引]
    ns: [[tid用户索引, ns_tid用户索引， 随机tid用户索引]，
         [ns_tid用户索引， tid用户索引, 随机tid用户索引]，
         [ns_tid用户索引， 随机tid用户索引, tid用户索引]]
    """
    tid1, tid2, ps1, ps2, ns1, ns2 = [], [], [], [], [], []
    node = []
    edge = [[], []]
    edge_attr = []
    global_spatial = []
    global_temporal = []

    tids = [dic['tid'] for dic in batch]

    for dic in batch:    # dic[key] = (node, edge_ind, edge_attr) or None
        add_len = len(node)
        node = node + dic['g1'][0]
        edge0, edge1 = dic['g1'][1]
        edge[0] += [i + add_len for i in edge0]
        edge[1] += [i + add_len for i in edge1]
        edge_attr.extend(dic['g1'][2])
        global_spatial += dic['g1'][3]
        global_temporal += dic['g1'][4]
        tid1.append(add_len)
        if dic['ps1'] is not None:
            add_len = len(node)
            node = node + dic['ps1'][0]
            edge0, edge1 = dic['ps1'][1]
            edge[0] += [i + add_len for i in edge0]
            edge[1] += [i + add_len for i in edge1]
            global_spatial += dic['ps1'][3]
            global_temporal += dic['ps1'][4]
            edge_attr.extend(dic['ps1'][2])
        ps1.append(add_len)

        add_len = len(node)
        node = node + dic['g2'][0]
        edge0, edge1 = dic['g2'][1]
        edge[0] += [i + add_len for i in edge0]
        edge[1] += [i + add_len for i in edge1]
        edge_attr.extend(dic['g2'][2])
        global_spatial += dic['g2'][3]
        global_temporal += dic['g2'][4]
        tid2.append(add_len)
        if dic['ps2'] is not None:
            add_len = len(node)
            node = node + dic['ps2'][0]
            edge0, edge1 = dic['ps2'][1]
            edge[0] += [i + add_len for i in edge0]
            edge[1] += [i + add_len for i in edge1]
            global_spatial += dic['ps2'][3]
            global_temporal += dic['ps2'][4]
            edge_attr.extend(dic['ps2'][2])
        ps2.append(add_len)

        if dic['ns1']:
            lis = []
            for ns in dic['ns1']:
                add_len = len(node)
                node = node + ns[0]
                edge0, edge1 = ns[1]
                edge[0] += [i + add_len for i in edge0]
                edge[1] += [i + add_len for i in edge1]
                edge_attr.extend(ns[2])
                global_spatial += ns[3]
                global_temporal += ns[4]
                lis.append(add_len)
            ns1.append(lis)
        else:
            ns1.append([])

        if dic['ns2']:
            lis = []
            for ns in dic['ns2']:
                add_len = len(node)
                node = node + ns[0]
                edge0, edge1 = ns[1]
                edge[0] += [i + add_len for i in edge0]
                edge[1] += [i + add_len for i in edge1]
                edge_attr.extend(ns[2])
                global_spatial += ns[3]
                global_temporal += ns[4]
                lis.append(add_len)
            ns2.append(lis)
        else:
            ns2.append([])

    # 随机选择tid
    for k, v in enumerate(ns1):
        add_num = len(batch) - len(v) if v else len(batch)
        if add_num == 0:
            continue
        if add_num == len(batch):
            add_ind = tid1.copy()
            add_ind.remove(tid1[k])
            add_ind.append(add_ind[0])
        else:
            add_ind = sample(tid1, add_num + 1)
            if tid1[k] in add_ind:
                add_ind.remove(tid1[k])
            else:
                add_ind = add_ind[: add_num]
        ns_list = ns1[k] + add_ind
        ns1[k] = ns_list

    for k, v in enumerate(ns2):
        add_num = len(batch) - len(v) if v else len(batch)
        if add_num == 0:
            continue
        if add_num == len(batch):
            add_ind = tid2.copy()
            add_ind.remove(tid2[k])
            add_ind.append(add_ind[0])
        else:
            add_ind = sample(tid2, add_num + 1)
            if tid2[k] in add_ind:
                add_ind.remove(tid2[k])
            else:
                add_ind = add_ind[: add_num]
        ns_list = ns2[k] + add_ind
        ns2[k] = ns_list

    for i in range(len(batch)):
        ns1[i][i] = tid1[i]
        ns1[i] = ns1[i][: len(batch)]

    for i in range(len(batch)):
        try:
            ns2[i][i] = tid2[i]
        except BaseException as e:
            print(i, len(tid2), len(ns2), len(ns2[i]))
            raise e
        ns2[i] = ns2[i][: len(batch)]

    node = torch.tensor(node, dtype=torch.float32)
    edge = torch.tensor(edge, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    global_spatial = torch.tensor(global_spatial, dtype=torch.float32)
    global_temporal = torch.tensor(global_temporal, dtype=torch.float32)
    return node, edge, edge_attr, global_spatial, global_temporal, tid1, tid2, ps1, ps2, ns1, ns2, tids


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
