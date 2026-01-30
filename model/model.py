import os
import torch
import torch.nn as nn
import torch.optim as optim

import dgl
import torch.nn.functional as F

from TransformerLayer import CNNTransformerEncoder

from torch_geometric.nn import HeteroConv, SAGEConv
from dgl.nn.pytorch import GINConv
from torch.nn import MultiheadAttention

class HeteroGNN(nn.Module):
    def __init__(self, drug_input_dim=50, protein_input_dim=50, hidden_dim=128, num_layers=2):
        super().__init__()

        self.drug_proj = nn.Linear(drug_input_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('drug', 'interacts', 'protein'): SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim
                ),
                ('protein', 'rev_interacts', 'drug'): SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim
                )
            }, aggr='mean')
            self.convs.append(conv)

        self.drug_output = nn.Linear(hidden_dim, 64)
        self.protein_output = nn.Linear(hidden_dim, 64)

    def forward(self, hetero_data, drug_indices, protein_indices):

        x_dict = {
            'drug': F.relu(self.drug_proj(hetero_data['drug'].x)),
            'protein': F.relu(self.protein_proj(hetero_data['protein'].x))
        }


        edge_index_dict = {
            ('drug', 'interacts', 'protein'): hetero_data['drug', 'interacts', 'protein'].edge_index,
            ('protein', 'rev_interacts', 'drug'): hetero_data['protein', 'rev_interacts', 'drug'].edge_index
        }

        for conv in self.convs:

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}


        drug_feats = self.drug_output(x_dict['drug'])
        protein_feats = self.protein_output(x_dict['protein'])


        batch_drug_feats = drug_feats[drug_indices]
        batch_protein_feats = protein_feats[protein_indices]

        return batch_drug_feats, batch_protein_feats

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.fc4 = nn.Linear(out_dim, binary)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        x=self.sigmoid(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, drug_dim, target_dim, output_dim):
        super().__init__()
        self.num_layers = 2


        self.drug_to_target_attn = nn.ModuleList([
            MultiheadAttention(embed_dim=drug_dim, kdim=target_dim, vdim=target_dim, num_heads=4)
            for _ in range(self.num_layers)
        ])


        self.target_to_drug_attn = nn.ModuleList([
            MultiheadAttention(embed_dim=target_dim, kdim=drug_dim, vdim=drug_dim, num_heads=4)
            for _ in range(self.num_layers)
        ])


        self.gate = nn.Linear(drug_dim + target_dim, 2)

        self.fc1 = nn.Linear(in_features=drug_dim + target_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.norm = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, drug, target):

        drug_residual = drug.mean(dim=1)
        target_residual = target.mean(dim=1)

        drug_as_q = drug.transpose(0, 1)
        target_as_kv = target.transpose(0, 1)

        for i in range(self.num_layers):

            drug_out, _ = self.drug_to_target_attn[i](drug_as_q, target_as_kv, target_as_kv)
            drug_as_q = drug_out

        drug_attn_out = drug_out.transpose(0, 1)

        target_as_q = target.transpose(0, 1)
        drug_as_kv = drug.transpose(0, 1)

        for i in range(self.num_layers):

            target_out, _ = self.target_to_drug_attn[i](target_as_q, drug_as_kv, drug_as_kv)
            target_as_q = target_out

        target_attn_out = target_out.transpose(0, 1)


        drug_pooled = drug_attn_out.mean(dim=1)
        target_pooled = target_attn_out.mean(dim=1)


        gate_input = torch.cat([drug_pooled, target_pooled], dim=-1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)

        drug_fused = gate_weights[:, 0:1] * drug_residual + drug_pooled
        target_fused = gate_weights[:, 1:2] * target_residual + target_pooled

        combined = torch.cat([drug_fused, target_fused], dim=-1)

        combined = self.dropout(combined)
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        combined = self.fc2(combined)

        return self.norm(combined)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return x + self.fc(x)


class Net(nn.Module):
    def __init__(self, device, **kwargs):
        super(Net, self).__init__()
        self.device = device
        self.dropout_val = kwargs.get('dropout', 0.2)

        self.hetero_gnn = HeteroGNN(drug_input_dim=50, protein_input_dim=50, hidden_dim=128, num_layers=2)
        self.hetero_data = kwargs.get('hetero_graph')['data'].to(device) if kwargs.get('hetero_graph') else None
        d_model = 32
        self.transformer = CNNTransformerEncoder(vocab_size=475, d_model=d_model, n_heads=4, n_layers=2, d_ff=64,
                                                 max_seq_length=800)
        self.protein_conv = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.drug_net = nn.Sequential(
            nn.Linear(kwargs.get('drug_len', 2048), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_val)
        )

        def gin_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        self.gnn_layers = nn.ModuleList([
            GINConv(gin_mlp(74, 64), 'sum'),
            GINConv(gin_mlp(64, 64), 'sum'),
            GINConv(gin_mlp(64, 64), 'sum')
        ])

        self.drug_proj_final = nn.Linear(640, 128)

        self.protein_proj_final = nn.Linear(192, 128)

        self.cross_modal_attention = CrossModalAttention(128, 128, 64)
        self.final_output = nn.Sequential(
            nn.Linear(64, 64),
            ResidualBlock(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=kwargs.get('learning_rate', 0.0001),
                                    weight_decay=kwargs.get('decay', 0.0))
        self.criterion = nn.BCELoss()

    def forward_gnn(self, g):
        h = g.ndata['h'].to(self.device)
        for i, layer in enumerate(self.gnn_layers):
            h = layer(g, h)
            h = F.relu(h).flatten(1) if i < len(self.gnn_layers) - 1 else h.mean(1)
        g.ndata['h_out'] = h
        return dgl.readout_nodes(g, 'h_out', op='mean')

    def forward(self, inputs):
        d_feat, _, d_kge, _, p_seq, _, _, d_graph, d_idx, p_idx = inputs

        s_d_hetero, s_p_hetero = self.hetero_gnn(self.hetero_data, d_idx, p_idx)

        p_trans = self.transformer(p_seq).permute(0, 2, 1)

        p_conv = self.protein_conv(p_trans).squeeze(-1)

        d_morgan = self.drug_net(d_feat)

        d_gnn = self.forward_gnn(d_graph)

        f_drug = F.relu(self.drug_proj_final(torch.cat([d_morgan, d_gnn, s_d_hetero], dim=1)))
        f_prot = F.relu(self.protein_proj_final(torch.cat([p_conv, s_p_hetero], dim=1)))


        combined = self.cross_modal_attention(f_drug.unsqueeze(1), f_prot.unsqueeze(1))

        return self.final_output(combined)

