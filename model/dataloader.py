import json
import pandas as pd
import numpy as np
import torch
from subword_nmt.apply_bpe import BPE
from torch_geometric.data import HeteroData
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import dgl

drug_col = 'DrugID'
protein_col = 'ProteinID'
label_col = 'Label'

vocab_csv = pd.read_csv('vocab.csv')
values = vocab_csv['Values'].values
protein_dict = dict(zip(values, range(1, len(values) + 1)))

csv1 = pd.read_csv('data/train/morgan_train.csv')
csv2 = pd.read_csv('data/valid/morgan_valid.csv')
csv3 = pd.read_csv('data/test/morgan_test.csv')
final = []
for k in range(len(csv1)):
    l = [i for i in csv1['SMILES'][k]]
    final += l
for k in range(len(csv2)):
    l = [i for i in csv2['SMILES'][k]]
    final += l
for k in range(len(csv3)):
    l = [i for i in csv3['SMILES'][k]]
    final += l
kk = list(set(final))
kk_dict = {k: v + 1 for v, k in enumerate(kk)}


def encod_SMILES(seq, kk_dict):
    if pd.isnull(seq):
        return [0]
    else:
        return [kk_dict[a] for a in seq]

vocab_txt = open('vocab.txt')
bpe = BPE(vocab_txt, merges=-1, separator='')

def encodeSeq(seq, protein_dict):
    firststep = bpe.process_line(seq).split()
    return [protein_dict[a] for a in firststep]

def pad_sequences(sequences, maxlen, padding='post', value=0):

    seq_lengths = [len(seq) for seq in sequences]
    if maxlen is None:
        maxlen = max(seq_lengths)

    padded_seqs = torch.full((len(sequences), maxlen), value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if padding == 'post':
                padded_seqs[i] = torch.tensor(seq[:maxlen], dtype=torch.long)
            else:
                padded_seqs[i] = torch.tensor(seq[-maxlen:], dtype=torch.long)
        else:
            if padding == 'post':
                padded_seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            else:
                padded_seqs[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
    return padded_seqs


def data(dti_dir, drug_dir, protein_dir, prot_vec='Convolution', prot_len=800,
               drug_vec='Convolution', drug_len=2048, drug_len2=100):
    print(f"parsing data: {dti_dir}, {drug_dir}, {protein_dir}")
    df=pd.read_csv(dti_dir)
    dti_df = pd.read_csv(dti_dir)

    kge_model = 'complex'
    embeddings_key = 'ent_embeddings.weight'
    if kge_model == 'complex' or kge_model == 'analogy':
        embeddings_key = 'ent_re_embeddings.weight'

    with open(f'data/kge/{kge_model}/entity_kge.vec', 'r') as f:
        entity_kge = f.readline()
        entity_kge = json.loads(entity_kge)
        entity_kge = entity_kge[embeddings_key]

    entity_kge_df = pd.DataFrame({'entity_kge': entity_kge})
    entity_all_df = pd.read_csv('./data/entity2id.txt', sep='\t', header=None,
                                names=['entity', 'entity_id'])
    entity_all_df['entity_kge'] = entity_kge_df
    entity_all_df = entity_all_df.drop(['entity_id'], axis=1)

    drug_kge = entity_all_df[entity_all_df['entity'].str.contains('D')]
    protein_kge = entity_all_df[entity_all_df['entity'].str.contains('P')]

    drug_kge.columns = ['DrugID', 'drug_kge']
    protein_kge.columns = ['ProteinID', 'protein_kge']

    drug_kge.set_index('DrugID', inplace=True)
    protein_kge.set_index('ProteinID', inplace=True)

    dti_df = pd.merge(dti_df, drug_kge, left_on=drug_col, right_index=True)
    dti_df = pd.merge(dti_df, protein_kge, left_on=protein_col, right_index=True)

    drug_df = pd.read_csv(drug_dir, index_col=drug_col)
    drug_df['drug_embedding'] = drug_df.SMILES.map(lambda a: encod_SMILES(a, kk_dict))

    protein_df = pd.read_csv(protein_dir, index_col=protein_col)
    protein_df['encoded_sequence'] = protein_df.Target_Sequence.map(lambda a: encodeSeq(a, protein_dict))

    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)

    c_train = dti_df['morgan_fp'].values
    l = []
    for i in c_train:
        temp = [int(k) for k in i]
        l.append(temp)
    drug_feature = np.array(l, dtype=np.float32)

    drug_feature2 = pad_sequences(dti_df['drug_embedding'].values, drug_len2, padding='post')


    drug_feature3 = torch.stack([torch.tensor(np.array(i), dtype=torch.float32)
                                 for i in dti_df['drug_kge'].values])


    protein_feature = pad_sequences(dti_df['encoded_sequence'].values, prot_len, padding='post')
    protein_feature2 = pad_sequences(dti_df['encoded_sequence'].values, prot_len, padding='post')
    protein_feature3 = torch.stack([torch.tensor(np.array(i), dtype=torch.float32)
                                    for i in dti_df['protein_kge'].values])

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    dti_df['dgl_graph'] = dti_df['SMILES_x'].apply(
        lambda s: dgl.add_self_loop(smiles_to_bigraph(
            s,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            explicit_hydrogens=False
        )) if pd.notnull(s) else None
    )


    invalid_idx = dti_df[dti_df['dgl_graph'].isnull()].index
    if len(invalid_idx) > 0:
        print(f"Warning: {len(invalid_idx)} invalid SMILES removed.")
        dti_df.drop(invalid_idx, inplace=True)

    g = dti_df['dgl_graph'][0]
    print(f"Number of nodes: {g.number_of_nodes()}")

    label = torch.tensor([int(i) for i in dti_df[label_col].values], dtype=torch.float32)

    unique_drug_ids = sorted(set(dti_df[dti_df['DrugID'].str.startswith('D')]['DrugID']))
    unique_protein_ids = sorted(set(dti_df[dti_df['ProteinID'].str.startswith('P')]['ProteinID']))

    drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(unique_drug_ids)}
    protein_id_to_idx = {protein_id: idx for idx, protein_id in enumerate(unique_protein_ids)}

    dti_df['drug_idx'] = dti_df['DrugID'].map(drug_id_to_idx)
    dti_df['protein_idx'] = dti_df['ProteinID'].map(protein_id_to_idx)
    data = HeteroData()

    drug_features = torch.stack([
        torch.tensor(dti_df[dti_df['DrugID'] == drug_id]['drug_kge'].iloc[0], dtype=torch.float32)
        for drug_id in unique_drug_ids
    ])
    data['drug'].x = drug_features  #

    protein_features = torch.stack([
        torch.tensor(dti_df[dti_df['ProteinID'] == protein_id]['protein_kge'].iloc[0], dtype=torch.float32)
        for protein_id in unique_protein_ids
    ])
    data['protein'].x = protein_features

    edge_index = torch.tensor([
        dti_df['DrugID'].map(drug_id_to_idx).values,
        dti_df['ProteinID'].map(protein_id_to_idx).values
    ], dtype=torch.long)
    data['drug', 'interacts', 'protein'].edge_index = edge_index
    data['drug', 'interacts', 'protein'].edge_label = torch.FloatTensor(dti_df['Label'].values)
    data['protein', 'rev_interacts', 'drug'].edge_index = edge_index.flip([0])

    data.drug_id_to_idx = drug_id_to_idx
    data.protein_id_to_idx = protein_id_to_idx


