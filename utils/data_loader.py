import os
import numpy as np
import dgl
import tqdm
import torch
import pickle
# from sklearn.model_selection import KFold
from utils.KFold import KFold, tvSplit
import sklearn.model_selection


def load_data(data_path='./data', device=torch.device('cpu')):
    if os.path.exists(os.path.join(data_path, 'kg_data.pkl')):
        with open(os.path.join(data_path, 'kg_data.pkl'), 'rb') as save_data:
            g, e_feat = pickle.load(save_data)
    else:
        # g
        # e_feat:整个网络中每条边的类型，是个list，对应于g中存储的边的顺序
        # nodes_list:整个网络中每种，每个node的feature
        edges = []
        edge_type = {}
        e_feat = []
        with open(os.path.join(data_path, 'edges.tsv'), 'r') as file:
            for line in tqdm.tqdm(file.readlines(), desc='loading kg'):
                h, r, t = line.strip().split('\t')
                [h, r, t] = [int(h), int(r), int(t)]
                if h == t:
                    continue
                edges.append([h, t])
                edges.append([t, h])
                e_feat.append(r + 1)
                e_feat.append(r + 1)
                # edge_type[(h, t)] = r + 1
                # edge_type[(t, h)] = r + 1

        g = dgl.graph(edges)
        g = dgl.remove_self_loop(g)  # 消除自环
        g = dgl.add_self_loop(g)  # 给孤立的点加自环

        for i in range(len(e_feat), g.num_edges()):
            e_feat.append(0)
        # for node in g.nodes():
        #     edge_type[(node.item(), node.item())] = 0
        # e_feat = []
        # head, tail = g.edges()
        # for i in tqdm.tqdm(range(len(head)), desc='loading kg edge type'):
        #     e_feat.append(edge_type[(head[i].item(), tail[i].item())])

        with open(os.path.join(data_path, 'kg_data.pkl'), 'wb') as file:
            pickle.dump([g, e_feat], file)

    smiles_list = []
    with open(os.path.join(data_path, 'smiles.tsv'), 'r') as file:
        for line in file.readlines():
            smiles = line.strip().split('\t')[1]
            smiles_list.append(smiles)

    nodes_list = []
    with open(os.path.join(data_path, 'nodes.tsv'), 'r') as file:
        for line in file.readlines():
            i, node_name, node_type = line.strip().split('\t')
            nodes_list.append([int(i), int(node_type)])

    edge_type_num = max(e_feat) + 1
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    nodes_list = torch.tensor(nodes_list).to(device)
    return g.to(device), e_feat, nodes_list, smiles_list, edge_type_num


def get_train_test(data_path='./data', fold_num=5, multi_class=False):
    sample = []
    with open(os.path.join(data_path, 'ddi.tsv'), 'r') as file:
        for line in file.readlines():
            d1, d2, label = [int(_) for _ in line.strip().split('\t')]
            sample.append([d1, d2, label])
    sample = np.array(sample)

    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, up_sample=multi_class)
    train_sample = []
    test_sample = []
    for train, test in kfold.split(sample):
        train_sample.append(sample[train])
        test_sample.append(sample[test])

    if not multi_class:
        drug = np.unique(sample[:, 0:2])
        ddi = set()
        with open(os.path.join(data_path, 'ddi.tsv'), 'r') as file:
            for line in file.readlines():
                line = [int(_) for _ in line.strip().split('\t')][:2]
                ddi.add(tuple(line))
        for fold in range(fold_num):
            train_sample[fold][:, 2] = 1
            test_sample[fold][:, 2] = 1

            sample_neg = []
            for i in range(len(train_sample[fold]) + len(test_sample[fold])):
                while True:
                    d1 = drug[np.random.randint(len(drug))]
                    d2 = drug[np.random.randint(len(drug))]
                    if (d1, d2) not in ddi and (d2, d1) not in ddi:
                        sample_neg.append([d1, d2, 0])
                        break
            train_sample_neg = sample_neg[:len(train_sample[fold])]
            test_sample_neg = sample_neg[len(train_sample[fold]):]

            train_sample[fold] = np.concatenate([train_sample[fold], train_sample_neg])
            test_sample[fold] = np.concatenate([test_sample[fold], test_sample_neg])

    train_sample = np.array(train_sample)
    test_sample = np.array(test_sample)
    return train_sample, test_sample


def get_train_valid_test(data_path='./data', fold_num=5, split_ratio=[8, 1], multi_class=False):
    sample = []
    with open(os.path.join(data_path, 'ddi.tsv'), 'r') as file:
        for line in file.readlines():
            d1, d2, label = [int(_) for _ in line.strip().split('\t')]
            sample.append([d1, d2, label])
    sample = np.array(sample)

    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, up_sample=False)
    tvsplit = tvSplit(split_ratio, shuffle=True, random_state=42, up_sample=False)
    train_sample = []
    valid_sample = []
    test_sample = []
    for train, valid_test in kfold.split(sample):
        valid, test = tvsplit.split(sample[valid_test])
        train_sample.append(sample[train])
        valid_sample.append(valid)
        test_sample.append(test)

    if not multi_class:
        drug = np.unique(sample[:, 0:2])
        ddi = set()
        with open(os.path.join(data_path, 'ddi.tsv'), 'r') as file:
            for line in file.readlines():
                line = [int(_) for _ in line.strip().split('\t')][:2]
                ddi.add(tuple(line))
        for fold in range(fold_num):
            train_sample[fold][:, 2] = 1
            valid_sample[fold][:, 2] = 1
            test_sample[fold][:, 2] = 1

            sample_neg = []
            for i in range(len(train_sample[fold]) + len(valid_sample[fold]) + len(test_sample[fold])):
                while True:
                    d1 = drug[np.random.randint(len(drug))]
                    d2 = drug[np.random.randint(len(drug))]
                    if (d1, d2) not in ddi and (d2, d1) not in ddi:
                        sample_neg.append([d1, d2, 0])
                        break
            train_sample_neg = sample_neg[:len(train_sample[fold])]
            valid_sample_neg = sample_neg[len(train_sample[fold]):len(train_sample[fold])+len(valid_sample[fold])]
            test_sample_neg = sample_neg[len(train_sample[fold])+len(valid_sample[fold]):]

            train_sample[fold] = np.concatenate([train_sample[fold], train_sample_neg])
            valid_sample[fold] = np.concatenate([valid_sample[fold], valid_sample_neg])
            test_sample[fold] = np.concatenate([test_sample[fold], test_sample_neg])

    train_sample = np.array(train_sample)
    valid_sample = np.array(valid_sample)
    test_sample = np.array(test_sample)
    return train_sample, valid_sample, test_sample


if __name__ == '__main__':
    os.chdir('../')
    # load_data()
    get_train_valid_test(multi_class=False)
    # get_train_test()
