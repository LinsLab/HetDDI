import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model.HetDDI import HetDDI
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger
import os

def run(args):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    kg_g, smiles = load_data(data_path, device=device)
    train_sample, test_sample = get_train_test(data_path, fold_num=args.fold_num,
                                               label_type=args.label_type, condition=args.condition)

    scores = []
    for i in range(0, args.fold_num):
        # for i in range(0, 1):
        # load data
        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        if args.label_type == 'multi_class':
            train_y = torch.from_numpy(train_y).long()
            test_y = torch.from_numpy(test_y).long()
        else:
            train_y = torch.from_numpy(train_y).float()
            test_y = torch.from_numpy(test_y).float()

        # load model
        if args.label_type == 'multi_class':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 86, args.condition).to(device)
            loss_func = nn.CrossEntropyLoss()
        elif args.label_type == 'binary_class':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 1, args.condition).to(device)
            loss_func = nn.BCEWithLogitsLoss()
        elif args.label_type == 'multi_label':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 200, args.condition).to(device)
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
                            i, epoch, args.batch_size, args.label_type, device)

            test_score = test(model, loss_func, test_x_left, test_x_right, test_y, i, epoch, args.batch_size,
                              args.label_type, device)

            test_acc = test_score[0]
            if epoch > 50:
                early_stopping(test_acc, model)
                if early_stopping.counter == 0:
                    best_test_score = test_score
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break
            # one epoch end
            print(best_test_score)
            print("=" * 100)

        # one fold end
        scores.append(best_test_score)
        print('Test set score:', scores)

    # all fold end, output the final result
    scores = np.array(scores)
    scores = scores.mean(axis=0)
    if args.label_type == 'multi_class':
        mean_kappa = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, kappa:{:.3f}\033[0m"
            .format(
                scores[0], scores[1], scores[2], scores[3], scores[4]
            ))
    elif args.label_type == 'binary_class':
        mean_auc = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, auc:{:.3f}\033[0m"
            .format(
                scores[0], scores[1], scores[2], scores[3], scores[4]
            ))
    elif args.label_type == 'multi_label':
        print(scores)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--batch_size', type=int, default=2 ** 15)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--hidden_dim', type=int, default=300, help='Dimension of the node hidden state. Default is 300.')
    ap.add_argument('--num_layer', type=int, default=3)
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)

    ap.add_argument('--label_type', type=str, choices=['multi_class', 'binary_class', 'multi_label'],
                    default='binary_class')
    ap.add_argument('--condition', type=str, choices=['s1', 's2', 's3'], default='s1')
    ap.add_argument('--mode', type=str, choices=['only_kg', 'only_mol', 'concat'], default='concat')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--kg_name', type=str, default='DRKG')
    ap.add_argument('--ddi_name', type=str, choices=['DrugBank', "TWOSIDES"], default='DrugBank')

    args = ap.parse_args(args=[])
    print(args)

    terminal = sys.stdout
    log_file = './log/ddi-dataset_{} label-type_{} mode_{} condition_{}.txt'. \
        format(args.hidden_dim, args.label_type, args.mode,args.condition)
    sys.stdout = Logger(log_file, terminal)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('running on', device)

    run(args)

