import os

import torch
import torch.nn as nn
import dgl
from model.hgnn import HGNN
from rdkit import Chem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, WeightAndSum, SumPooling

class Mol(nn.Module):
    def __init__(self, smiles, num_hidden, num_layer, device='cuda:0', condition='s1'):
        super(Mol, self).__init__()
        self.device = device

        self.readout = AvgPooling()

        mol_g = graph_construction(smiles)
        self.mol_g = dgl.batch(mol_g).to(self.device)

        nodes_type = self.mol_g.ndata['atomic_number'].tolist()
        nodes = []
        for i in range(len(nodes_type)):
            if condition == 's1':
                nodes.append([i, nodes_type[i]])
                # nodes.append([nodes_type[i], nodes_type[i]])
            else:
                nodes.append([nodes_type[i], 0])
                # nodes.append([nodes_type[i], nodes_type[i]])
        nodes = torch.tensor(nodes).to(self.device)

        self.gnn = HGNN(self.mol_g, self.mol_g.edata['bond_type'], nodes, num_hidden, num_layer).to(device)
        if not condition == 's1':
            self.gnn.node_embedding = nn.Embedding(self.gnn.nodes[:, 0].max() + 2, num_hidden)
            self.gnn.load_state_dict(torch.load('./mol_weight.pth'))

    def forward(self):
        result = self.readout(self.mol_g, self.gnn())
        return result

def graph_construction(smiles):
    graphs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        g = mol_to_bigraph(mol, add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=False)
        graphs.append(g)

    return graphs
