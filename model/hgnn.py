import torch
import torch.nn as nn
import torch.nn.functional as F
from model.conv import HetConv


class HGNN(nn.Module):
    def __init__(self,
                 g,
                 e_feat,
                 edge_type_num,
                 nodes,
                 num_hidden,
                 num_layer=3
                 ):
        super(HGNN, self).__init__()
        self.g = g
        self.e_feat = e_feat

        self.nodes = nodes
        self.num_hidden = num_hidden

        self.node_embedding = nn.Embedding(self.nodes[:, 0].max() + 1, num_hidden)
        nn.init.kaiming_normal_(self.node_embedding.weight)

        self.edge_types = torch.unique(e_feat).sort().values
        self.edge_dim = 16
        self.edge_embedding = nn.Embedding(self.e_feat.max() + 1, self.edge_dim)

        self.num_etypes = edge_type_num
        self.num_layer = num_layer
        self.heads = 1
        self.heads = [1] + [self.heads] * (self.num_layer - 1) + [1]
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        self.feat_drop = 0.5
        self.attn_drop = 0.5
        self.negative_slope = 0.02
        self.residual = True
        self.bn = False
        self.alpha = 0

        for l in range(self.num_layer):
            if l == self.num_layer - 1:
                activation = None
            else:
                activation = self.activation
            self.gat_layers.append(HetConv(self.edge_dim, self.num_etypes, self.nodes[:, 1],
                                           self.num_hidden * self.heads[l], self.num_hidden, self.heads[l + 1],
                                           self.feat_drop, self.attn_drop, self.negative_slope,
                                           self.residual,
                                           activation, bn=self.bn,
                                           alpha=self.alpha))

    def get_output_size(self):
        # return (self.num_layer + 1) * self.num_hidden
        return self.num_hidden

    def forward(self):
        # 将所有node的feature通过各自feature的FC层转置为h，然后拼接在一起后经过l2正则
        h = self.node_embedding(self.nodes[:, 0])

        res_attn = None
        for l in range(self.num_layer):
            h, res_attn = self.gat_layers[l](self.g, h, self.e_feat, res_attn=res_attn)
            h = h.mean(1)

        return h
