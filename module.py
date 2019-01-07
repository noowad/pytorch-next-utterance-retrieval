import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_idx, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
    def forward(self, x):
        return self.embedding(x)


class BOWEncoder(nn.Module):
    def __init__(self, hidden_size, embedding):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        sent_len = x.size(2)
        context_len = x.size(1)
        batch_size = x.size(0)
        x = x.view(batch_size * context_len, -1)  # (bs*context_len, sent_len)
        embedding = self.embedding(x)
        embedding = embedding.view(batch_size, context_len, sent_len, -1)  # (bs, context_len, sent_len, embed_size)
        encoded = torch.sum(self.linear(embedding), dim=2)
        return encoded


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding):
        super().__init__()
        self.hidden_size = hidden_size // 2
        self.embedding = embedding
        self.gru = nn.GRU(input_size, self.hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        embed = self.embedding(x)
        output, pre_hidden = self.gru(embed.transpose(0, 1))
        hidden = pre_hidden.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        return hidden
