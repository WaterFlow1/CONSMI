# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.utils.rnn as rnn_utils


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_embd = 256
    n_head = 8
    n_layer = 8
    resid_pdrop = 0.1
    num_props = 0
    scaffold_maxlen = 48

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


config = GPTConfig(95, 100)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props)) + int(
            config.scaffold_maxlen)  # int(config.lstm_layers)    #  int(config.scaffold)
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                             .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # self.attn = CausalSelfAttention(config)
        self.attn = MultiHeadAttention()

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x, attn_mask):
        xx = self.ln1(x)
        # y, attn = self.attn(xx)
        y, attn = self.attn(xx, xx, xx, attn_mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_K = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_V = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.fc = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, config.n_head, 32).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, config.n_head, 32).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, config.n_head, 32).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, config.n_head, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, config.n_embd)
        output = self.fc(context)
        return nn.LayerNorm(config.n_embd).to('cuda:3')(output + residual), attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(32)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class TextEmbedding(nn.Module):
    def __init__(self, src_vocab_size, d_model=256):
        super(TextEmbedding, self).__init__()
        # self.input_d_channel_size = [81,128,256,512]
        # self.filter_d_size = [32, 32, 32, 32]
        self.tok_emb = nn.Embedding(95, 256, padding_idx=16)
        self.type_emb = nn.Embedding(2, 256)
        self.pos_emb = nn.Parameter(torch.zeros(1, 111, 256))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(8)])
        self.projection = nn.Sequential(
            nn.Linear(d_model, 2048, bias=False),
            nn.LeakyReLU(),
            nn.Linear(2048, src_vocab_size, bias=False),
        )

    def forward(self, enc_inputs):
        b, t = enc_inputs.shape[0], enc_inputs.shape[1]
        attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        token_embeddings = self.tok_emb(enc_inputs)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=enc_inputs.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        for layer in self.blocks:
            x, attn = layer(x, attn_mask)
        logits = self.projection(x)
        if torch.isnan(logits).any():
            print(logits)
        return logits




def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(16).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class Encoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers,
                 bidirectional, dropout, latent_size):
        super(Encoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(30,
                                  768, 3,
                                  batch_first=True, dropout=0.2,
                                  )
        # self.linear_layer = nn.Linear(
        #     (int(bidirectional) + 1) * num_layers * hidden_size,
        #     latent_size
        # )

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)

        return x


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, outputs, labels):
        batch_size, vocab_size, dim = outputs.shape

        outputs = outputs.view(batch_size * vocab_size, dim)
        outputs = F.normalize(outputs, p=2, dim=1)

        similarity_matrix = torch.matmul(outputs, outputs.T) / self.temperature

        labels = labels.unsqueeze(1).expand(-1, vocab_size).reshape(-1)

        mask = torch.eye(batch_size * vocab_size, device=outputs.device).bool()
        labels_eq = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        labels_eq.masked_fill_(mask, False)

        positive_indices = labels_eq.fill_diagonal_(0)

        positive_log_probs = torch.log_softmax(similarity_matrix, dim=1)[positive_indices]

        loss = -positive_log_probs.mean()

        return loss
