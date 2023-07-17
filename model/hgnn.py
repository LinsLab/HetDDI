import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from torch.utils.checkpoint import checkpoint

class HetConv(nn.Module):
    def __init__(self, nodes, edges, num_hidden, activation=F.elu, batch_norm=True, negative_slope=0.02):
        super(HetConv, self).__init__()

        self.nodes = nodes
        self.edges = torch.arange(0, edges.max() + 1).to(nodes.device)
        # edge_dim = 16
        edge_dim = num_hidden
        self.edge_embedding = nn.Embedding(edges.max() + 1, edge_dim)

        if batch_norm:
            # self.bn = nn.BatchNorm1d(num_hidden)
            self.bn = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.BatchNorm1d(num_hidden)
            )
        else:
            self.bn = None
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.nodes_fc = nn.Parameter(torch.FloatTensor(size=(nodes[:, 1].max() + 1, num_hidden)))
        # self.nodes_fc = nn.Parameter(torch.FloatTensor(size=(1, num_hidden)))
        # self.nodes_fc = self.nodes_fc[self.nodes[:, 1]]
        self.edges_fc = nn.Parameter(torch.FloatTensor(size=(edges.max() + 1, edge_dim)))
        self.nodes_attn = nn.Parameter(torch.FloatTensor(size=(1, num_hidden)))
        self.edges_attn = nn.Parameter(torch.FloatTensor(size=(1, edge_dim)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        nn.init.xavier_uniform_(self.nodes_fc)
        nn.init.xavier_uniform_(self.edges_fc)
        nn.init.xavier_uniform_(self.nodes_attn)
        nn.init.xavier_uniform_(self.edges_attn)

    def forward(self, g, nodes_feat, edges_feat):
        g = g.local_var()
        nodes_feat = nodes_feat * self.nodes_fc[0]
        # nodes_feat = nodes_feat * self.nodes_fc[self.nodes[:, 1]]
        g.ndata.update({'feat': nodes_feat,
                        'ft': (nodes_feat * self.nodes_attn).sum(dim=-1)})
        g.apply_edges(fn.u_add_v('ft', 'ft', 'e'))

        all_edge_emb = self.edge_embedding(self.edges)
        all_edge_emb = all_edge_emb * self.edges_fc
        ee = (all_edge_emb * self.edges_attn).sum(dim=-1)[edges_feat]
        g.edata.update({'ee': ee})

        e = self.leaky_relu(g.edata.pop('e') + g.edata.pop('ee'))
        # g.edata.update({'a': edge_softmax(g, e)})
        g.edata.update({'a': e})

        # message passing
        g.update_all(fn.u_mul_e('feat', 'a', 'm'), fn.sum('m', 'feat'))
        nodes_feat = g.ndata['feat']

        if self.bn:
            nodes_feat = self.bn(nodes_feat)
        if self.activation:
            nodes_feat = self.activation(nodes_feat)
        return nodes_feat

class HGNN(nn.Module):
    def __init__(self,
                 g,
                 edges,
                 nodes,
                 num_hidden,
                 num_layer=3
                 ):
        super(HGNN, self).__init__()
        self.g = g
        self.nodes = nodes
        self.edges = edges

        self.num_layer = num_layer
        self.num_hidden = num_hidden

        self.residual = True
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Identity()
        self.bn = True

        self.node_embedding = nn.Embedding(self.nodes[:, 0].max() + 2, num_hidden)

        self.gat_layers = nn.ModuleList()
        for l in range(self.num_layer):
            self.gat_layers.append(HetConv(nodes, edges, num_hidden, activation=F.relu, batch_norm=self.bn))
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)

    def init_emb(self, emb):
        self.node_embedding = self.node_embedding.from_pretrained(emb.float(), freeze=True)

    def get_output_size(self):
        # return (self.num_layer + 1) * self.num_hidden
        return self.num_hidden

    def forward(self):
        all_layer_node_feats = [self.node_embedding(self.nodes[:, 0])]

        for l in range(self.num_layer):
            node_feats = self.gat_layers[l](self.g, all_layer_node_feats[-1], self.edges)
            if self.residual:
                node_feats = node_feats + all_layer_node_feats[-1]
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        return all_layer_node_feats[-1]
