import math

import torch
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    # def split_heads(self, x, batch_size):
    #     """Split the last dimension into (num_heads, depth).
    #     Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    #     """
    #     x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    #     return tf.transpose(x, perm=[0, 2, 1, 3])
    def scaled_dot_product_attention(self, q, k, v, mask, adjoin_matrix):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        # dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / math.sqrt(k.size(k)[-1])

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        if adjoin_matrix is not None:
            scaled_attention_logits += adjoin_matrix

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attn = self.softmax(scaled_attention_logits).to(q.dtype)
        drop_attn = self.dropout(attn)

        output = torch.matmul(drop_attn, v)

        # attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        # output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, drop_attn

    def forward(self, v, k, q, mask, adjoin_matrix):
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        dim_per_head = self.depth
        head_count = self.num_heads

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        q = shape(q)
        k = shape(k)
        v = shape(v)
        # q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        # k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix)

        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, training, mask, adjoin_matrix):
        attn_output, attention_weights = self.mha(x, x, x, mask, adjoin_matrix)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        out1 = x + attn_output

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                         self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training, mask, adjoin_matrix):
        seq_len = x.shape[1]
        adjoin_matrix.unsqueeze(1)
        # adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        # adding embedding and position encoding.
        x = self.embedding(x)   # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.d_model)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention_weights = self.enc_layers[i](x, training, mask, adjoin_matrix)
        return x  # (batch_size, input_seq_len, d_model)


class BertModel(nn.Module):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17, dropout_rate=0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff, input_vocab_size=vocab_size, maximum_position_encoding=200,
                               rate=dropout_rate)
        self.fc1 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc2 = nn.Linear(d_model, vocab_size)

    def forward(self, x, adjoin_matrix, mask, training=False):
        x = self.encoder(x, training=training, mask=mask, adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x





