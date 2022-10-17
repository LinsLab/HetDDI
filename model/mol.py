import os

import torch
import torch.nn as nn
import dgl
from model.hgnn import HGNN
from rdkit import Chem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, WeightAndSum,SumPooling


class Mol(nn.Module):
    def __init__(self, smiles, num_hidden, num_layer, device='cuda:0'):
        super(Mol, self).__init__()
        self.device = device

        self.readout = AvgPooling()
        # self.readout = SumPooling()
        # self.readout = MaxPooling()
        # self.readout = WeightAndSum(num_hidden)

        mol_g, success = graph_construction_and_featurization(smiles)
        for i in range(len(mol_g)):
            mol_g[i] = mol_g[i]
        self.mol_g = dgl.batch(mol_g).to(self.device)

        edges_type = self.mol_g.edata['bond_type'].to(self.device)
        edge_type_num = edges_type.max()+1
        # edge_type_num = 1
        nodes_type = self.mol_g.ndata['atomic_number'].tolist()
        nodes = []
        for i in range(len(nodes_type)):
            # nodes.append([nodes_type[i], nodes_type[i]])
            # nodes.append([i, 0])
            nodes.append([i, nodes_type[i]])
        nodes = torch.tensor(nodes).to(self.device)

        self.gnn = HGNN(self.mol_g, edges_type, edge_type_num, nodes, num_hidden, num_layer).to(device)

    def forward(self):
        # nfeats = [self.mol_g.ndata['atomic_number'].to(self.device),
        #           self.mol_g.ndata['chirality_type'].to(self.device)]
        # efeats = [self.mol_g.edata['bond_type'].to(self.device),
        #           self.mol_g.edata['bond_direction_type'].to(self.device)]
        # return self.readout(self.mol_g, self.GIN(self.mol_g, nfeats, efeats))
        return self.readout(self.mol_g, self.gnn())


def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) 应该等同于
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization 不成功，捕捉错误，在尝试一次
    # 捕捉错误的步骤
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_construction_and_featurization(smiles):
    """Construct graphs from SMILES and featurize them
    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation
    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    # print(len(smiles))
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            # mol = molecule_from_smiles(smi)
            # print(mol is None)
            if mol is None:
                success.append(False)
                continue
            # print(mol)
            # print('it can go there.')
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            # print('it can also go there.')
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)
    # print(len(graphs))

    return graphs, success

# if __name__ == '__main__':
#     os.chdir('../')
#     model = Mol(['C[N+](C)(C)CCO', 'N[C@@H](CCC(N)=O)C(O)=O', 'CSCC[C@H](N)C(O)=O'], device='cpu')
#     model()
#     print(model())