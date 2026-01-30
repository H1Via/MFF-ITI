import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn = self.scaled_dot_product_attention(Q, K, V, mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CNNFeatureExtractor(nn.Module):


    def __init__(self, input_dim, d_model, kernel_sizes=[3,5,7], num_filters=64):
        super(CNNFeatureExtractor, self).__init__()
        self.d_model = d_model


        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])


        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        self.feature_weights = nn.Parameter(torch.ones(len(kernel_sizes)))


        self.projection = nn.Linear(num_filters * len(kernel_sizes), d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        x = x.transpose(1, 2)

        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))
            conv_outputs.append(conv_out)

        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)
        output = self.projection(combined)
        return self.dropout(output)


class CNNTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, use_cnn=True, input_dim=None):
        super(CNNTransformerEncoderLayer, self).__init__()
        self.use_cnn = use_cnn

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        if use_cnn and input_dim:
            self.cnn_extractor = CNNFeatureExtractor(input_dim, d_model)
        else:
            self.cnn_extractor = None

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) if use_cnn else None

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None):

        if self.use_cnn and self.cnn_extractor:
            cnn_features = self.cnn_extractor(src)

            src = self.norm3(src + self.dropout(cnn_features))


        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        if self.use_cnn:
            cnn_output = self.cnn_extractor(src)
            src = self.norm3(src + self.dropout(cnn_output))

        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src


class CNNTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length,
                 use_cnn=True, input_dim=None):
        super(CNNTransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size+1, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            CNNTransformerEncoderLayer(
                d_model, n_heads, d_ff,
                use_cnn=use_cnn and i<2,

                input_dim = d_model if use_cnn and i < 2 else None
            )
            for i in range(n_layers)
        ])

        self.dropout = nn.Dropout(0.1)

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src):
        src_mask = self.generate_mask(src)

        src_embedded = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_embedded = self.positional_encoding(src_embedded)
        src_embedded = self.dropout(src_embedded)

        for encoder_layer in self.encoder_layers:
            src_embedded = encoder_layer(src_embedded, src_mask)
        return src_embedded




