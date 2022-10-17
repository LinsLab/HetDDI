import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model.myModel import myModel
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger


def run(args):
    np.random.seed(42)
    torch.manual_seed(42)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(42)  # 为当前GPU设置

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # load data
    '''
        g: 知识图谱网络，去除掉了86种DDI
        features_list: 整个网络中每种，每个node的feature
        e_feat: 整个网络中每条边的类型，是个list，对应于g中存储的边的顺序
        mol_emb: 86种DDI中，每个药物的分子式向量
    '''
    g, e_feat, nodes_list, mol_emb, edge_type_num = load_data(args.data_path, device=device)
    # e_feat = torch.zeros(e_feat.shape).to(device, dtype=torch.int)
    # edge_type_num = 1
    train_sample, test_sample = get_train_test(args.data_path, fold_num=args.fold_num, multi_class=args.multi_class)

    scores = []
    for i in range(0 , args.fold_num):
        # load data
        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2]


        # load model
        if args.multi_class:
            model = myModel(g, e_feat, edge_type_num, nodes_list, mol_emb, args.hidden_dim, args.num_layer, args.mode
                            , 86).to(device)
            loss_func = nn.CrossEntropyLoss()
        else:
            model = myModel(g, e_feat, edge_type_num, nodes_list, mol_emb, args.hidden_dim, args.num_layer, args.mode
                            , 1).to(device)
            loss_func = nn.BCEWithLogitsLoss()
        if i == 0:
            print(model)

        # divide parameters into two parts, weight_p has l2_norm but bias_bn_emb_p not
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name or 'bn' in name or 'embedding' in name:
                bias_bn_emb_p += [p]
            else:
                weight_p += [p]
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ]
        # train setting
        optimizer = optim.Adam(model_parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_test_score = None
        for epoch in range(args.epoch):
            train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                  i, epoch, args.batch_size, args.multi_class, device)

            test_score = test(model, loss_func, test_x_left, test_x_right, test_y, i, epoch, args.batch_size,
                              args.multi_class, device)

            test_f1 = test_score[1]
            if epoch > 0:
                early_stopping(test_f1, model)
                if early_stopping.counter == 0:
                    best_test_score = test_score
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    scores.append(best_test_score)
                    break
            # one epoch end
            print(best_test_score)
            print("=" * 100)

        # one fold end
        print('Test set score:', scores)

    # all fold end, output the final result
    scores = np.array(scores)
    mean_acc = scores[:, 0].mean()
    mean_f1 = scores[:, 1].mean()
    mean_precision = scores[:, 2].mean()
    mean_recall = scores[:, 3].mean()
    if args.multi_class:
        mean_kappa = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, kappa:{:.3f}\033[0m"
                .format(
                mean_acc, mean_f1, mean_precision, mean_recall, mean_kappa
            ))
    else:
        mean_auc = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, auc:{:.3f}\033[0m"
                .format(
                mean_acc, mean_f1, mean_precision, mean_recall, mean_auc
            ))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--hidden_dim', type=int, default=300, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_layer', type=int, default=3)
    ap.add_argument('--epoch', type=int, default=400, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=20, help='Patience.')
    ap.add_argument('--batch_size', type=int, default=2 ** 15)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    # ap.add_argument('--weight-decay', type=float, default=0)

    ap.add_argument('--multi_class', type=bool, default=True)
    ap.add_argument('--mode', type=str, choices=['only_kg', 'only_mol', 'concat'], default='concat')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--kg_path', type=str, default='./data/DRKG')
    ap.add_argument('--ddi_path', type=str, default='./data/DrugBank')

    args = ap.parse_args(args=[])

    terminal = sys.stdout
    log_file = './log/hidden-dim_{} multi-class_{} mode_{} 2.txt'. \
        format(args.hidden_dim, args.multi_class, args.mode)
    sys.stdout = Logger(log_file, terminal)

    print(args)
    run(args)

    sys.stdout.end()
