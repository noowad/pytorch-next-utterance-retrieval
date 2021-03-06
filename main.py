import torch
import torch.nn as nn
from module import *
import config
import model
import fire
import train
from utils import *
from tokenization import MecabTokenizer, TokenIndexing, Tokenizing
from data import DialogueDataset, DataLoader, make_vocab


def main(mode='train',
         data_dir='datas',
         vocab='datas/vocab.txt',
         word_embed='datas/aka_fasttext_20181029.vec',
         save_dir='persona_model',
         max_len=100,
         model_file=None):
    cfg = config.TrainConfig
    model_cfg = config.ModelConfig
    print('Data Loading...')
    if vocab is None:
        vocab = make_vocab(data_dir)
    set_seeds(cfg.seed)
    tokenizer = MecabTokenizer(vocab, model_cfg.min_cnt)
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                TokenIndexing(vocab, DialogueDataset.labels, model_cfg.max_context_len, max_len, model_cfg.min_cnt)]
    dataset = DialogueDataset(data_dir, mode, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print('Model Loading...')
    persona_model = model.PersonaBasedModel(model_cfg, word_embed)
    criterion = nn.BCELoss()
    trainer = train.Trainer(cfg,
                            persona_model,
                            data_iter,
                            torch.optim.Adam(persona_model.parameters(), lr=cfg.lr),
                            save_dir, get_device())

    if mode == 'train':
        print('mode:train')

        def get_loss(model, batch, global_step):  # make sure loss is a scalar tensor
            contexts_ids, query_ids, response_ids, label_ids = batch
            logits = model(contexts_ids, query_ids, response_ids)
            label_ids = label_ids.type(torch.float).unsqueeze(1)
            loss = criterion(logits, label_ids)
            output = (logits.squeeze() > 0.5).float()
            result = (output == label_ids.squeeze()).float()
            acc = result.mean()
            return loss, acc

        trainer.train(get_loss)
    elif mode == 'eval':
        print('mode:eval')

        def evaluate(model, batch):
            contexts_ids, query_ids, response_ids, label_ids = batch
            logits = model(contexts_ids, query_ids, response_ids)
            label_ids = label_ids.type(torch.float).unsqueeze(1)
            output = (logits.squeeze() > 0.5).float()
            result = (output == label_ids.squeeze()).float()
            acc = result.mean()
            return acc, result

        results = trainer.eval(evaluate, model_file)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
