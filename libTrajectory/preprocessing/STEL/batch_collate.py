from random import sample

import torch


def collate_fun(batch):
    # data[key] = (node, edge_ind, edge_attr) or None
    tid1, tid2, ps1, ps2, ns1, ns2 = [], [], [], [], [], []
    # g1
    node1 = []
    edge_ind1 = [[], []]
    edge_attr1 = []
    # g2
    node2 = []
    edge_ind2 = [[], []]
    edge_attr2 = []

    for ind, dic in enumerate(batch):
        add_len1 = len(node1)
        node1 = node1 + dic['g1'][0]
        e_ind0, e_ind1 = dic['g1'][1]
        edge_ind1[0] += [i + add_len1 for i in e_ind0]
        edge_ind1[1] += [i + add_len1 for i in e_ind1]
        edge_attr1.extend(dic['g1'][2])
        tid1.append(add_len1)

        add_len2 = len(node2)
        node2 = node2 + dic['g2'][0]
        e_ind0, e_ind1 = dic['g2'][1]
        edge_ind2[0] += [i + add_len2 for i in e_ind0]
        edge_ind2[1] += [i + add_len2 for i in e_ind1]
        edge_attr2.extend(dic['g2'][2])
        tid2.append(add_len2)

        if dic['ps1'] is not None:
            add_len1 = len(node1)
            node1 = node1 + dic['ps1'][0]
            e_ind0, e_ind1 = dic['ps1'][1]
            edge_ind1[0] += [i + add_len1 for i in e_ind0]
            edge_ind1[1] += [i + add_len1 for i in e_ind1]
            edge_attr1.extend(dic['ps1'][2])
        ps1.append(add_len1)

        if dic['ps2'] is not None:
            add_len2 = len(node2)
            node2 = node2 + dic['ps2'][0]
            e_ind0, e_ind1 = dic['ps2'][1]
            edge_ind2[0] += [i + add_len2 for i in e_ind0]
            edge_ind2[1] += [i + add_len2 for i in e_ind1]
            edge_attr2.extend(dic['ps2'][2])
        ps2.append(add_len2)

        if dic['ns1'] is not None:
            add_len1 = len(node1)
            node1 = node1 + dic['ns1'][0]
            e_ind0, e_ind1 = dic['ns1'][1]
            edge_ind1[0] += [i + add_len1 for i in e_ind0]
            edge_ind1[1] += [i + add_len1 for i in e_ind1]
            edge_attr1.extend(dic['ns1'][2])
            ns1.append(add_len1)
        else:
            ns1.append(None)  # 后续从tid1随机选择

        if dic['ns2'] is not None:
            add_len2 = len(node2)
            node2 = node2 + dic['ns2'][0]
            e_ind0, e_ind1 = dic['ns2'][1]
            edge_ind2[0] += [i + add_len2 for i in e_ind0]
            edge_ind2[1] += [i + add_len2 for i in e_ind1]
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
    edge_ind1 = torch.tensor(edge_ind1, dtype=torch.long)
    edge_attr1 = torch.tensor(edge_attr1, dtype=torch.float32)
    node2 = torch.tensor(node2, dtype=torch.float32)
    edge_ind2 = torch.tensor(edge_ind2, dtype=torch.long)
    edge_attr2 = torch.tensor(edge_attr2, dtype=torch.float32)
    return node1, edge_ind1, edge_attr1, node2, edge_ind2, edge_attr2, tid1, tid2, ps1, ps2, ns1, ns2