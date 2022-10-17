"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


# pylint: enable=W0235
class HetConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 edge_feats,  # 边的feature长度
                 num_etypes,  # 边的数量
                 node_types,  # 所有节点类型
                 in_feats,  # node的输入feature(X)长度
                 out_feats,  # 输出的长度
                 num_heads,  # 头数
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,  # LeakyReLU的负斜率角度
                 residual=True,  #
                 activation=None,  # 用于更新后的节点的激活函数，用的elu
                 allow_zero_in_degree=False,
                 bias=False,
                 bn=False,
                 alpha=0.):
        super(HetConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)  # 将in_feats复制为两份
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.edge_types = torch.tensor(range(num_etypes)).to(node_types.device)
        self.edge_embedding = nn.Embedding(num_etypes, edge_feats)

        self.node_types = node_types
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            # self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            num_ntypes = node_types.max() + 1
            self.fc = nn.Parameter(th.FloatTensor(size=(num_ntypes, num_heads, out_feats)))
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

        self.bn = bn
        if bn:
            # self.bn = nn.Sequential(
            #     nn.Linear(out_feats, out_feats),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(out_feats),
            #     nn.Dropout(0.1)
            # )
            self.bn = nn.BatchNorm1d(out_feats)


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            # nn.init.xavier_normal_(self.fc.weight, gain=gain)
            nn.init.xavier_normal_(self.fc, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():  # 只应用graph，而不改变其中的值
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                # h_src = self.feat_drop(feat)
                # h_dst = self.feat_drop(feat)
                h_src = h_dst = feat
                # feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                h_src = h_src.view(-1, self._num_heads, self._out_feats)
                h_dst = h_dst.view(-1, self._num_heads, self._out_feats)
                feat_src = (h_src * self.fc[self.node_types])
                feat_dst = (h_dst * self.fc[self.node_types])
                feat_src = self.feat_drop(feat_src)
                feat_dst = self.feat_drop(feat_dst)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            all_edge_emb = self.edge_embedding(self.edge_types)
            all_edge_emb = self.fc_e(all_edge_emb).view(-1, self._num_heads, self._edge_feats)
            ee = (all_edge_emb * self.attn_e).sum(dim=-1).unsqueeze(-1)[e_feat]
            # ee = torch.full((e_feat.shape[0],1,1),
            #                 (all_edge_emb * self.attn_e).sum(dim=-1).unsqueeze(-1)[0][0][0]).to(e_feat.device)
            graph.edata.update({'ee': ee})

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))
            # e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1 - self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.bn:
                for i in range(rst[0].shape[0]):
                    rst[:, i] = self.bn(rst[:, i])
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()
