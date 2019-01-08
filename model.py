import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from module import *


class PersonaBasedModel(nn.Module):
    def __init__(self, cfg, word_embed=None):
        super().__init__()
        self.is_temp_enc = cfg.is_temp_enc
        self.encoder_type = cfg.encoder_type
        self.embedding = WordEmbedding(cfg.vocab_size, cfg.embed_size, 0, word_embed)
        if self.encoder_type == 'gru':
            self.query_encoder = GRUEncoder(cfg.embed_size, cfg.encoder_size, self.embedding)
            self.response_encoder = GRUEncoder(cfg.embed_size, cfg.encoder_size, self.embedding)
        self.input_context_encoder = BOWEncoder(cfg.encoder_size, self.embedding)
        self.output_context_encoder = BOWEncoder(cfg.encoder_size, self.embedding)
        self.temp_enc_for_m = nn.Parameter(torch.Tensor(1, cfg.max_context_len, cfg.embed_size).normal_(0, 0.1))
        self.temp_enc_for_c = nn.Parameter(torch.Tensor(1, cfg.max_context_len, cfg.embed_size).normal_(0, 0.1))
        self.classifier = nn.Linear(cfg.embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, contexts, query, response):
        # modules
        query_u = self.query_encoder(query)
        response = self.response_encoder(response)
        context_m = self.input_context_encoder(contexts)
        context_c = self.output_context_encoder(contexts)
        batch_size = context_m.size(0)
        memory_len = context_m.size(1)
        if self.is_temp_enc:
            context_m += self.temp_enc_for_m.repeat(batch_size, 1, 1)[:, :memory_len, :]
            context_c += self.temp_enc_for_c.repeat(batch_size, 1, 1)[:, :memory_len, :]
        prob = F.softmax((context_m * query_u.unsqueeze(1)).sum(2), dim=1)  # (batch_size,memory_len)
        o = (prob.unsqueeze(2) * context_c).sum(1)  # (batch_size,embed_size)
        o = (o + query_u) * response  # residual connection
        logits = self.sigmoid(self.classifier(o))
        return logits
