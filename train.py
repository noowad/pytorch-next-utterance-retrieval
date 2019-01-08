from tqdm import tqdm
import torch
import torch.nn as nn
import os


class Trainer(object):
    """Training Helper Class"""

    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg  # config for training : see class Config
        self.model = model
        self.data_iter = data_iter  # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device  # device name

    def train(self, get_loss, data_parallel=True):
        """ Train Loop """
        self.model.train()  # train mode
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        print('start training...')
        global_step = 0  # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            acc_sum = 0.
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss, acc = get_loss(model, batch, global_step)  # mean() for Data Parallelism
                loss = loss.mean()
                acc = acc.mean()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                acc_sum += acc.item()
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

                if global_step % self.cfg.save_steps == 0:  # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f Average Acc %5.3f ' % (
                        e + 1, self.cfg.n_epochs, loss_sum / (i + 1), acc_sum / (i + 1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step)  # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f Average Acc %5.3f' % (
                e + 1, self.cfg.n_epochs, loss_sum / (i + 1), acc_sum / (i + 1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval()  # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = []  # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # evaluation without gradient calculation
                accuracy, result = evaluate(model, batch)  # accuracy to print
            results.append(result)

            iter_bar.set_description('Iter(acc=%5.3f)' % accuracy)
        return results

    def load(self, model_file):
        """ load saved persona_model or pretrained transformer (a part of persona_model) """
        if model_file:
            print('Loading the persona_model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

    def save(self, i):
        """ save current persona_model """
        torch.save(self.model.state_dict(),  # save persona_model object before nn.DataParallel
                   os.path.join(self.save_dir, 'model_steps_' + str(i) + '.pt'))
