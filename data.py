from torch.utils.data import Dataset, DataLoader
import csv
import torch
import itertools
import os
import collections
from utils import convert_to_unicode
import MeCab


class DialogueDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = ('neg', 'pos')

    def __init__(self, dir, mode, pipeline=[]):  # csv file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(os.path.join(dir, mode + '.csv'), "r") as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines):  # instance : tuple of fields
                for proc in pipeline:  # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):  # skip header
            yield line[-1], line[0].split(' '), line[1], line[2]  # label, contexts, query, answer


def load_vocab(vocab_file, min_cnt):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0

    with open(vocab_file, "r") as reader:
        while True:
            line = convert_to_unicode(reader.readline())
            if not line:
                break
            token = line.split('\t')[0]
            cnt = int(line.split('\t')[1])
            if cnt >= min_cnt:
                vocab[token] = index
                index += 1
    return vocab


def make_vocab(data_dir):
    print("making vocab...")
    counts = collections.Counter()
    tagger = MeCab.Tagger("-Owakati")
    for line in open(os.path.join(data_dir, 'train.csv')).read().splitlines():
        utterances = line.split('\t')[:-1]
        for utterance in utterances:
            tokens = tagger.parse(utterance).split()
            counts.update([str(token).lower() for token in tokens])
    vocabs = sorted(counts.items(), key=lambda t: t[1], reverse=True)
    with open(os.path.join(data_dir, 'vocab.txt'), 'w') as fout:
        fout.write('<PAD>' + '\t10000000\n')
        fout.write('<UNK>' + '\t10000000\n')
        fout.write('<START>' + '\t10000000\n')
        fout.write('<END>' + '\t10000000\n')
        for word, freq in vocabs:
            fout.write(word + '\t' + str(freq) + '\n')
    return os.path.join(data_dir, 'vocab.txt')
