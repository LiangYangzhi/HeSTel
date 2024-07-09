from random import sample
import torch


def train_coll(batch):
    """
    tid: [tid1用户索引, tid2用户索引, tid3用户索引]
    ps: [[ps_tid1用户索引, ps_tid1用户索引， ps_tid1用户索引],
        [ps_tid2用户索引, ps_tid2用户索引， ps_tid2户索引],
        [ps_tid3用户索引, ps_tid3用户索引， ps_tid3用户索引]]
    ns: [[tid1用户索引, ns_tid用户索引， 随机tid用户索引]，
         [ns_tid用户索引， tid2用户索引, 随机tid用户索引]，
         [ns_tid用户索引， 随机tid用户索引, tid3用户索引]]
    """
    tids = [dic['tid'] for dic in batch]
    tid1, tid2, ps1, ps2, ns1, ns2 = [], [], [], [], [], []
    node = []
    edge = [[], []]
    edge_attr = []
    global_spatial = []
    global_temporal = []

    for dic in batch:    # dic[key] = (node, edge_ind, edge_attr) or None
        for name in ['g1', 'g2']:
            add_len = len(node)
            node = node + dic[name][0]
            edge0, edge1 = dic[name][1]
            edge[0] += [i + add_len for i in edge0]
            edge[1] += [i + add_len for i in edge1]
            edge_attr.extend(dic[name][2])
            global_spatial += dic[name][3]
            global_temporal += dic[name][4]
            if name == "g1":
                tid1.append(add_len)
            elif name == "g2":
                tid2.append(add_len)

        for name in ['ps1', 'ps2', 'ns1', 'ns2']:
            lis = []
            for en_g in dic[name]:
                add_len = len(node)
                node = node + en_g[0]
                edge0, edge1 = en_g[1]
                edge[0] += [i + add_len for i in edge0]
                edge[1] += [i + add_len for i in edge1]
                global_spatial += en_g[3]
                global_temporal += en_g[4]
                edge_attr.extend(en_g[2])
                lis.append(add_len)
            exec(f"{name}.append(lis)")
    # ps1不足的，添加ps_tid1
    for k, v in enumerate(ps1):
        add_num = len(batch) - len(v) if v else len(batch)
        ps_list = ps1[k] + [tid1[k] for _ in range(add_num)]
        ps1[k] = ps_list

    # ps2不足的，添加ps_tid2
    for k, v in enumerate(ps2):
        add_num = len(batch) - len(v) if v else len(batch)
        ps_list = ps2[k] + [tid2[k] for _ in range(add_num)]
        ps2[k] = ps_list

    # 随机选择ns_tid1
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

    # 随机选择ns_tid2
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
        ns2[i][i] = tid2[i]
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
