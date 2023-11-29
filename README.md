# HetDDI

## Overview

This repository is the source code of our paper "HetDDI: a pre-trained heterogeneous graph neural network model for drug-drug interaction prediction".

## Environment Setting

This code is based on Pytorch and dgl-cuda. You need prepare your virtual enviroment early.

## Running the code

You can run the following command to run our work:

> python main.py

There are several parameters can be customized:

1. **batch_size**
2. **label_type**, you can choose one of "multi_class", "binary_class" or "multi_label". <br />"multi_class" and "binary_class" is only available at ddi_name = "DrugBank". <br />"multi_label" is only available at ddi_name = "TWOSIDES" 
3. **condition**, you can choose one of scenarios s1, s2, s3 defined in our paper, default is "s1"
4. **mode**, you can choose one of variants "HetDDI-mol", "HetDDI-kg" or "HetDDI" by "only_mol", "only_kg", "concat". Default is "HetDDI" by "concat".
5. **ddi_name**, the dataset you want to run, "DrugBank" or "TWOSIDES". Default is "DrugBank"
## Dataset Preparation

The dataset used in paper is available at **/HetDDI/data/DRKG+DrugBank** and **/HetDDI/data/DRKG+TWOSIDES**

If you want to use yourself dataset, you need to follow these format.

#### 1. nodes.tsv

the form is look like:

> 1618    Compound::DB09499  0

- node id    
- node name  
- node type id

#### 2. edges.tsv

the form is look like:

> 318	14	30460

- head node id    
- relation id
- tail node id

#### 3. smiles.tsv

the form is look like:

> 9   C[N+](C)(C)CCO

- node(drug) id
- smiles string

#### 4. ddi.tsv

the form is look like:

> 78  1616   59

- node(drug) id
- node(drug) id
- interaction type

## Weight files

The weight files for the model can be obtained from the following link.

The path for the weight files should be the root directory of project.

https://drive.google.com/drive/folders/1VKbVVzAcv_e3UgxId-Jrpac2SKqnCWeN



