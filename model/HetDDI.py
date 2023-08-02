import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from model.hgnn import HGNN
from model.decoder import Mlp
from model.mol import Mol
from torch.utils.checkpoint import checkpoint


class HetDDI(nn.Module):
    def __init__(self,
                 kg_g,
                 smiles,
                 num_hidden,
                 num_layer,
                 mode,
                 class_num,
                 condition
                 ):
        super(HetDDI, self).__init__()

        self.smiles = smiles
        self.device = kg_g.device
        self.mode = mode
        self.drug_num = len(smiles)

        dropout = 0.1
        if self.mode == 'only_kg' or self.mode == 'concat':
            self.kg = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], num_hidden, num_layer=num_layer)
            self.kg.load_state_dict(torch.load('./kg_weight.pth'))
            self.kg_size = self.kg.get_output_size()
            self.kg_fc = nn.Sequential(nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU(),

                                       nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU(),

                                       nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU()
                                       )

        if self.mode == 'only_mol' or self.mode == 'concat':
            self.mol = Mol(smiles, num_hidden, num_layer, self.device, condition)
            self.mol_size = self.mol.gnn.get_output_size()
            self.mol_fc = nn.Sequential(nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU(),

                                        nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU(),

                                        nn.Linear(self.mol_size, self.mol_size),
                                        nn.BatchNorm1d(self.mol_size),
                                        nn.Dropout(dropout),
                                        nn.ReLU()
                                        )

        if self.mode == 'only_kg':
            self.decoder = Mlp(self.kg_size, 0, class_num=class_num)
        elif self.mode == 'only_mol':
            self.decoder = Mlp(0, self.mol_size, class_num=class_num)
        else:
            self.decoder = Mlp(self.kg_size, self.mol_size, class_num=class_num)

    def forward(self, left, right):
        if self.mode == 'only_kg':
            # kg_emb = checkpoint(self.kg)[:self.drug_num]
            kg_emb = self.kg()[:self.drug_num]
            kg_emb = self.kg_fc(kg_emb)

            left_kg_emb = kg_emb[left]
            right_kg_emb = kg_emb[right]

            return self.decoder(left_kg_emb, right_kg_emb)

        elif self.mode == 'only_mol':
            mol_emb = self.mol()
            mol_emb = self.mol_fc(mol_emb)

            left_mol_emb = mol_emb[left]
            right_mol_emb = mol_emb[right]

            return self.decoder(left_mol_emb, right_mol_emb)

        else:
            # kg_emb = checkpoint(self.kg)[:self.drug_num]
            kg_emb = self.kg()[:self.drug_num]
            kg_emb = self.kg_fc(kg_emb)

            left_kg_emb = kg_emb[left]
            right_kg_emb = kg_emb[right]

            mol_emb = self.mol()
            mol_emb = self.mol_fc(mol_emb)

            left_mol_emb = mol_emb[left]
            right_mol_emb = mol_emb[right]

            left_emb = torch.concat([left_kg_emb, left_mol_emb], dim=-1)
            right_emb = torch.concat([right_kg_emb, right_mol_emb], dim=-1)

            return self.decoder(left_emb, right_emb)
