import os


def prcess_data(kg_path='./data/DRKG', ddi_path='./data/DrugBank', output_path='./data'):
    all_ddi = set()
    all_ddi_compound = set()
    with open(os.path.join(ddi_path, 'DDI.txt'), 'r') as file:
        for line in file.readlines():
            d1, d2 = line.strip().split('\t')[:2]
            all_ddi.add((d1, d2))
            all_ddi.add((d2, d1))
            all_ddi_compound.add(d1)
            all_ddi_compound.add(d2)

    entities = set()
    relation = set()
    entity_type = set()
    kg = []
    with open(os.path.join(kg_path, 'kg.tsv'), 'r') as file:
        for line in file.readlines():
            e1, r, e2 = line.strip().split('\t')
            e1_type = e1.split("::")[0]
            e2_type = e2.split("::")[0]
            entities.add(e1)
            entities.add(e2)
            entity_type.add(e1_type)
            entity_type.add(e2_type)
            # if e1_type == 'Compound' and e2_type == 'Compound':
            #     continue
            if (e1, e2) in all_ddi or (e2, e1) in all_ddi:
                continue
            # r = e1_type + ':' + e2_type
            relation.add(r)
            kg.append([e1, r, e2])

    with open(os.path.join(output_path, 'nodes.tsv'), 'w') as file:
        entities = entities.difference(all_ddi_compound)
        entity_type = list(entity_type)
        entity_type.sort()
        entity_type2index = {}
        entity2index = {}
        for i, e_type in enumerate(entity_type):
            entity_type2index[e_type] = i + 1
        all_ddi_compound = list(all_ddi_compound)
        all_ddi_compound.sort()
        for i, compound in enumerate(all_ddi_compound):
            file.write('\t'.join([str(i), compound, str(0)]) + '\n')
            entity2index[compound] = i
        entities = list(entities)
        entities.sort()
        for i, e in enumerate(entities):
            if e in all_ddi_compound:
                continue
            file.write('\t'.join([str(i + len(all_ddi_compound)), e, str(entity_type2index[e.split('::')[0]])]) + '\n')
            entity2index[e] = i + len(all_ddi_compound)

    with open(os.path.join(output_path, 'edges.tsv'), 'w') as file:
        relation = list(relation)
        relation.sort()
        relation2index = {}
        for i, r in enumerate(relation):
            relation2index[r] = i
        for item in kg:
            e1, r, e2 = item
            e1 = entity2index[e1]
            e2 = entity2index[e2]
            r = relation2index[r]
            file.write('\t'.join([str(e1), str(r), str(e2)])+'\n')

    with open(os.path.join(ddi_path, 'DDI.txt'), 'r') as file:
        with open(os.path.join(output_path, 'ddi.tsv'), 'w') as file_:
            for line in file.readlines():
                c1, c2, r = line.strip().split('\t')
                c1 = str(entity2index[c1])
                c2 = str(entity2index[c2])
                r = str(int(r)-1)

                file_.write('\t'.join([c1, c2, r])+'\n')

    with open(os.path.join(ddi_path, 'smiles.csv'), 'r') as file:
        with open(os.path.join(output_path, 'smiles.tsv'), 'w') as file_:
            for line in file.readlines()[1:]:
                c, smiles = line.strip().split(',')
                c = str(entity2index[c])

                file_.write('\t'.join([c, smiles])+'\n')

if __name__ == '__main__':
    os.chdir('../')
    prcess_data()
