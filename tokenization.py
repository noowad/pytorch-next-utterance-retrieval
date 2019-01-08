import collections
import os
import MeCab
from utils import convert_to_unicode
from data import load_vocab


class Pipeline():
    """ Preprocess Pipeline Class : callable """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """

    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor  # e.g. text normalization
        self.tokenize = tokenize  # tokenize function

    def __call__(self, instance):
        label, text_as, text_b, text_c = instance
        label = self.preprocessor(label)
        tokens_as = [self.tokenize(self.preprocessor(text_a)) for text_a in text_as]
        tokens_b = self.tokenize(self.preprocessor(text_b))
        tokens_c = self.tokenize(self.preprocessor(text_c)) if text_c else []

        return (label, tokens_as, tokens_b, tokens_c)


class MecabTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, min_cnt):
        self.vocab = load_vocab(vocab_file, min_cnt)

    def tokenize(self, text):
        tagger = MeCab.Tagger("-Owakati")
        return tagger.parse(text).split()

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)

    def convert_to_unicode(self, text):
        return convert_to_unicode(text)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """

    def __init__(self, vocab_file, labels, max_context_len, max_len, min_cnt):
        super().__init__()
        self.vocab_file = vocab_file
        self.indexer = convert_tokens_to_ids  # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_context_len = max_context_len
        self.max_len = max_len
        self.min_cnt = min_cnt

    def __call__(self, instance):
        label, tokens_as, tokens_b, tokens_c = instance
        context_ids = [self.indexer(load_vocab(self.vocab_file, self.min_cnt), tokens_a) for tokens_a in tokens_as]
        query_id = self.indexer(load_vocab(self.vocab_file, self.min_cnt), tokens_b)
        response_id = self.indexer(load_vocab(self.vocab_file, self.min_cnt), tokens_c)
        label_id = self.label_map[label]
        # zero padding
        for context_id in context_ids:
            context_id.extend([0] * (self.max_len - len(context_id)))
        for i in range(self.max_context_len - len(context_ids)):
            context_ids.append([0] * self.max_len)
        query_id.extend([0] * (self.max_len - len(query_id)))
        response_id.extend([0] * (self.max_len - len(response_id)))
        return (context_ids, query_id, response_id, label_id)


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        token = token.lower()
        ids.append(vocab.get(token, 1))
    return ids
