import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, kg_size, drug_size, class_num):
        super(Mlp, self).__init__()
        num_hidden = (kg_size + drug_size) * 2

        act = nn.ReLU()
        dropout = nn.Dropout(0.5)
        self.fc_layer = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                                      nn.BatchNorm1d(num_hidden),
                                      act,
                                      dropout,

                                      nn.Linear(num_hidden, num_hidden),
                                      nn.BatchNorm1d(num_hidden),
                                      act,
                                      dropout,

                                      nn.Linear(num_hidden, num_hidden),
                                      nn.BatchNorm1d(num_hidden),
                                      act,
                                      dropout
                                      )
        self.output_layer = nn.Sequential(nn.Linear(num_hidden, class_num, bias=False))

    def forward(self, *embs):
        input = torch.cat((embs), dim=-1)
        pred = self.fc_layer(input)
        pred = self.output_layer(pred)

        return pred
