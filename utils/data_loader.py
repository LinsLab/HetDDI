import os
import numpy as np
import dgl
import tqdm
import torch
from torch.utils import data
import pickle
# from sklearn.model_selection import KFold
from utils.KFold import KFold
import pandas as pd
import sklearn.model_selection


def load_data(data_path='./data', device=torch.device('cpu')):
    if os.path.exists(os.path.join(data_path, 'kg_data.pkl')):
        with open(os.path.join(data_path, 'kg_data.pkl'), 'rb') as save_data:
            kg_g, e_feat = pickle.load(save_data)
    else:
        # g
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

        kg_g = dgl.graph(edges)
        kg_g = dgl.remove_self_loop(kg_g)  # 消除自环
        kg_g = dgl.add_self_loop(kg_g)  # 给孤立的点加自环

        e_feat = np.concatenate([np.array(e_feat), np.zeros(kg_g.num_edges() - len(e_feat), ).astype(np.int64)])
        # for node in g.nodes():
        #     edge_type[(node.item(), node.item())] = 0
        # e_feat = []
        # head, tail = g.edges()
        # for i in tqdm.tqdm(range(len(head)), desc='loading kg edge type'):
        #     e_feat.append(edge_type[(head[i].item(), tail[i].item())])

        nodes_list = []
        with open(os.path.join(data_path, 'nodes.tsv'), 'r') as file:
            for line in file.readlines():
                i, node_name, node_type = line.strip().split('\t')
                nodes_list.append([int(i), int(node_type)])
        kg_g.ndata.update({'nodes': torch.tensor(nodes_list)})
        kg_g.edata.update({'edges': torch.from_numpy(e_feat)})

        with open(os.path.join(data_path, 'kg_data.pkl'), 'wb') as file:
            pickle.dump([kg_g, e_feat], file)

    smiles_list = []
    with open(os.path.join(data_path, 'smiles.tsv'), 'r') as file:
        for line in file.readlines():
            smiles = line.strip().split('\t')[1]
            smiles_list.append(smiles)

    return kg_g.to(device), smiles_list


def get_train_test(data_path='./data', fold_num=5, label_type='multi_class', condition='s1'):
    sample = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values

    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, up_sample=(label_type == 'multi_class'), condition=condition)
    train_sample = []
    test_sample = []
    for train, test in kfold.split(sample):
        train_sample.append(train)
        test_sample.append(test)

    # generate negative samples
    if label_type == 'binary_class' or label_type == 'multi_label':
        if label_type == 'multi_label':
            all_ddi = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values
        else:
            all_ddi = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values[:, :2]
        ddi = set()
        for item in all_ddi:
            ddi.add('\t'.join([str(_) for _ in item]))

        for fold in range(fold_num):
            if condition == 's1':
                train_drug = np.arange(max(train_sample[fold][:, :2].max(), test_sample[fold][:, :2].max())+1)
                test_drug = train_drug
            else:
                train_drug = np.unique(train_sample[fold][:, :2])
                test_drug = np.unique(test_sample[fold][:, :2])

            if label_type == 'binary_class':
                train_sample[fold][:, 2] = 1
                test_sample[fold][:, 2] = 1

            sample_neg = []
            sample_pos = np.concatenate([train_sample[fold], test_sample[fold]])
            for i in range(len(train_sample[fold])):
                while True:
                    d1 = train_drug[np.random.randint(len(train_drug))]
                    d2 = train_drug[np.random.randint(len(train_drug))]
                    if d1 == d2:
                        continue
                    if label_type == 'binary_class' and ('\t'.join([str(d1), str(d2)]) not in ddi and '\t'.join([str(d2), str(d1)]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
                    elif label_type == 'multi_label' and ('\t'.join([str(d1), str(d2), str(sample_pos[i][2])]) not in ddi and '\t'.join([str(d2), str(d1), str(sample_pos[i][2])]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
            for i in range(len(train_sample[fold]), len(train_sample[fold])+len(test_sample[fold])):
                while True:
                    d1 = test_drug[np.random.randint(len(test_drug))]
                    d2 = test_drug[np.random.randint(len(test_drug))]
                    if d1 == d2:
                        continue
                    if label_type == 'binary_class' and ('\t'.join([str(d1), str(d2)]) not in ddi and '\t'.join([str(d2), str(d1)]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
                    elif label_type == 'multi_label' and ('\t'.join([str(d1), str(d2), str(sample_pos[i][2])]) not in ddi and '\t'.join([str(d2), str(d1), str(sample_pos[i][2])]) not in ddi):
                        sample_neg.append([d1, d2, 0])
                        break
            train_sample_neg = np.array(sample_neg[:len(train_sample[fold])])
            test_sample_neg = np.array(sample_neg[len(train_sample[fold]):])
            if label_type == 'multi_label':
                train_sample[fold] = np.concatenate([train_sample[fold], train_sample[fold][:, 2:]], axis=-1)
                test_sample[fold] = np.concatenate([test_sample[fold], test_sample[fold][:, 2:]], axis=-1)
                train_sample[fold][:, 2] = 1
                test_sample[fold][:, 2] = 1

                train_sample_neg = np.concatenate([train_sample_neg, np.zeros((train_sample_neg.shape[0], 1), dtype=train_sample_neg.dtype)], axis=-1)
                test_sample_neg = np.concatenate([test_sample_neg, np.zeros((test_sample_neg.shape[0], 1), dtype=test_sample_neg.dtype)], axis=-1)
                train_sample_neg[:, 3] = train_sample[fold][:, 3]
                test_sample_neg[:, 3] = test_sample[fold][:, 3]

            train_sample[fold] = np.concatenate([train_sample[fold], train_sample_neg])
            test_sample[fold] = np.concatenate([test_sample[fold], test_sample_neg])

    return train_sample, test_sample

if __name__ == '__main__':
    os.chdir('../')
    # load_data()
    # get_train_test()
