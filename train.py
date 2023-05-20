#pylint: disable=missing-module-docstring
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
from ipdb import set_trace
import wandb
#  sys.path.insert(0, './')

from model import lcnn #pylint: disable=import-error
from dataset import dataset #pylint: disable=import-error
from pyutils import utils #pylint: disable=import-error
from model.loss import FocalLoss #pylint: disable=import-error
from model.loss import AutomaticWeightedLoss

class Trainer(): #pylint: disable=missing-class-docstring
    def __init__(self):
        utils.set_seed()
        self.parse_args()
        self.read_conf()
        self.log = os.path.join('exp', self.args['task'])
        self.log_interval = self.conf['log']['log_interval']
        self.metalambda = self.conf['train']['metalambda']
        self.grllambda = self.conf['train']['grllambda']

        trainset = dataset.MetaDataset(self.args['task'], data = 'train')
        self.args['collate_fn'] = self.conf['dataloader']['collate_fn']
        if self.conf['dataloader']['collate_fn'] == 'random':
            self.conf['dataloader']['collate_fn'] = trainset.random_collate_fn
        elif self.conf['dataloader']['collate_fn'] == 'balance':
            self.conf['dataloader']['collate_fn'] = trainset.balance_collate_fn
        self.trainloader = DataLoader(trainset, **self.conf['dataloader'])
        valset = dataset.CNSpoofDataset(self.args['task'], data = 'val')
        evalset = dataset.CNSpoofDataset(self.args['task'], data = 'test')
        self.valloader = DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 1)
        self.evalloader = DataLoader(evalset, batch_size = 1, shuffle = False, num_workers = 1)

        os.makedirs(self.log, exist_ok = True)
        self.device = torch.device(self.conf['train']['device'])
        self.model = lcnn.LCNN(**self.conf['model']['model_args'])
        self.model.to(self.device)
        self.adalossweight = AutomaticWeightedLoss(3)
        
        if self.conf['train']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                    [{'params': self.model.parameters()}, {'params': self.adalossweight.parameters(), 'weight_decay': 0.0}],
                    lr = 0.001, momentum = 0.9, weight_decay= 0.0001
                    )
        else:
            raise NotImplementedError("Only SGD optimizer")
        if self.conf['train']['scheduler'] == 'step':
            self.lr_scheduler = lr_scheduler.StepLR(
                    self.optimizer,
                    **self.conf['train']['scheduler_params']
                    )
        else:
            raise NotImplementedError("Only StepLR scheduler")

        self.criterion = nn.BCELoss()
        self.domain_criterion = FocalLoss(None, self.conf['train']['gamma'])
        self.epoch = self.conf['train']['epoch']
        self.start = 1
        print("Initialization done!")

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default = 'train', type = str)
        parser.add_argument('--task', default = 'ordinary', type = str)
        parser.add_argument('--batch-size', dest = 'batch_size', default = 32, type = int)
        parser.add_argument('--lr', default = 0.01, type = float)
        parser.add_argument('--momentum', default = 0.9, type = float)
        parser.add_argument('--weight-decay', dest = 'weight_decay', default = 0.0001, type = float)
        parser.add_argument('--step-size', dest = "step_size", default = 1, type = int)
        parser.add_argument('--gamma', default = 0.95, type = float)
        parser.add_argument('--ckpt', default = '', type = str)
        self.args = vars(parser.parse_args())

    def read_conf(self):
        f = open('conf/conf.yaml', 'r')
        self.conf = yaml.load(f, Loader = yaml.CLoader)
        f.close()

    def save(self, epoch):
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = self.lr_scheduler.state_dict()
        os.makedirs(os.path.join(self.log, 'models'), exist_ok = True)
        torch.save(
                {
                    'model': model_state_dict,
                    'optimizer': optimizer_state_dict,
                    'epoch': epoch,
                    'scheduler': scheduler_state_dict
                    },
                os.path.join(self.log, 'models', '{}.ckpt'.format(epoch))
                )

    def load(self, path):
        ckpt = torch.load(os.path.join(self.log, 'models/{}'.format(path)))
        model_state_dict = ckpt['model']
        optimizer_state_dict = ckpt['optimizer']
        scheduler_state_dict = ckpt['scheduler']
        self.start = ckpt['epoch']
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.lr_scheduler.load_state_dict(scheduler_state_dict)

    def split_model_parameters(self):
        model_params = []
        for name, p in self.model.named_parameters():
            if not 'frontend' in name and not 'alignment' in name:
                model_params.append(p)
        return model_params

    def train(self):
        wandb.init(
            project = 'interspeech2023',
            group = 'CM_genre_meta_learning_grl',
            job_type = self.args['task'],
            name = self.args['task'] + '-' + self.args['collate_fn'],
            config = self.conf #.update(self.args)
            )
        for epoch in range(self.start, self.epoch + 1):
            self.train_epoch(epoch)
            self.lr_scheduler.step()
            if epoch % self.log_interval == 0:
                self.eval(epoch, 'test')
                self.save(epoch)
        wandb.finish()

    def train_epoch(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.trainloader)
        sum_mtr_loss, sum_mtr_dloss, sum_mtr_samples, mtr_correct, mtr_dcorrect = 0, 0, 0, 0, 0
        sum_mte_loss, sum_mte_samples, mte_correct = 0, 0, 0
        for batch_idx, data in enumerate(progress_bar):
            sum_mtr_samples += len(data[0])
            sum_mte_samples += len(data[4])
            mtr_prediction, mtr_dprediction, mtr_loss, mtr_dloss, mte_prediction, mte_loss = self.train_batch(data)
            sum_mtr_loss += mtr_loss.item() * len(data[0])
            sum_mtr_dloss += mtr_dloss.item() * len(data[0])
            mtr_correct += mtr_prediction
            mtr_dcorrect += mtr_dprediction
            sum_mte_loss += mte_loss.item() * len(data[4])
            mte_correct += mte_prediction
            progress_bar.set_description(
                    'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] MTRL: {:.4f} MTRDL: {:.4f} MTRA: {:.4f}% MTRDA: {:.4f}% MTEL: {:.4f} MTEA: {:.4f}%'.format(
                        epoch, batch_idx + 1, len(self.trainloader),
                        100. * (batch_idx + 1) / len(self.trainloader),
                        sum_mtr_loss / sum_mtr_samples, sum_mtr_dloss / sum_mtr_samples,
                        100. * mtr_correct / sum_mtr_samples, 100. * mtr_dcorrect / sum_mtr_samples,
                        sum_mte_loss / sum_mte_samples,
                        100. * mte_correct / sum_mte_samples
                        )
                    )
            wandb.log(
                        {
                            'train/mtr_acc': mtr_correct / sum_mtr_samples,
                            'train/mtr_dacc': mtr_dcorrect / sum_mtr_samples,
                            'train/mte_acc': mte_correct / sum_mte_samples
                        }
                    )

    def train_batch(self, data): #pylint: disable=missing-function-docstring
        for weight in self.split_model_parameters():
            weight.fast = None
        mtr_audios, mtr_labels, mtr_lengths, mtr_genres, mte_audios, mte_labels, mte_lengths, mte_genres = data #pylint: disable=unused-variable
        # ----- Meta Training ------
        mtr_audios = mtr_audios.to(self.device)
        mtr_labels = mtr_labels.to(self.device)
        mtr_genres = mtr_genres.to(self.device)

        mtr_scores, mtr_dscores = self.model(mtr_audios, mtr_lengths)

        mtr_correct = self.compute_correct(mtr_scores, mtr_labels)
        mtr_dcorrect = self.compute_dcorrect(mtr_dscores, mtr_genres)

        mtr_loss = self.criterion(mtr_scores, mtr_labels)
        mtr_dloss = self.domain_criterion(mtr_dscores, mtr_genres)

        mtr_grads = torch.autograd.grad(mtr_loss, self.split_model_parameters(), create_graph = True, allow_unused=True)
        for k, weight in enumerate(self.split_model_parameters()):
            weight.fast = weight - self.optimizer.param_groups[0]['lr'] * mtr_grads[k]

        # ----- Meta Testing ------
        # loss is computed by using the \theta^' as intermediate variable
        mte_audios = mte_audios.to(self.device)
        mte_labels = mte_labels.to(self.device)
        mte_genres = mte_genres.to(self.device)

        #  set_trace()
        mte_scores, _ = self.model(mte_audios, mte_lengths)
        mte_correct = self.compute_correct(mte_scores, mte_labels)
        mte_loss = self.criterion(mte_scores, mte_labels)

        #  total_loss = self.metalambda * mte_loss + self.grllambda * mtr_dloss + (1 - self.metalambda - self.grllambda) * mtr_loss
        total_loss = self.adalossweight(mte_loss, mtr_dloss, mtr_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # ----- wandb log ------
        wandb.log(
                    {
                        'train/mtr_loss': mtr_loss,
                        'train/mtr_dloss': mtr_dloss,
                        'train/mte_loss': mte_loss,
                        'train/total_loss': total_loss,
                        'train/lr': self.optimizer.state_dict()['param_groups'][0]['lr']
                    }
                )
        return mtr_correct, mtr_dcorrect, mtr_loss, mtr_dloss, mte_correct, mte_loss

    def compute_correct(self, scores, labels):
        correct = ((scores > 0.5) == labels).sum().item()
        return correct

    def compute_dcorrect(self, scores, labels):
        predict = torch.argmax(scores, dim = 1)
        correct = (predict == labels).sum().item()
        return correct

    def eval(self, epoch, mode = 'test'):
        for weight in self.split_model_parameters():
            weight.fast = None
        if mode == 'test':
            dataloader = self.evalloader
        elif mode == 'val':
            dataloader = self.valloader
        else:
            raise NotImplementedError("Error")
        os.makedirs(os.path.join(self.log, 'scores'), exist_ok = True)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            with open(os.path.join(self.log, 'scores/{}_{}.score'.format(mode, epoch)), 'w') as f:
                for audio, label, genre, genreid, datalengths in tqdm(dataloader):
                    audio = audio.to(self.device)
                    label = label.to(self.device)
                    score, _ = self.model(audio, datalengths)
                    predict = score > 0.5
                    if predict == label:
                        correct += 1
                    line = str(score.item()) + ' ' + str(label.item()) + ' ' + genre[0] + '\n'
                    f.write(line)
        print('{} Acc: {:3.3f}%'.format(mode, 100 * correct / len(dataloader)))
        try:
            wandb.log({
                '{}/acc'.format(mode): correct / len(dataloader)
                })
        except:
            pass

if __name__ == '__main__':
    trainer = Trainer()
    if trainer.args['mode'] == 'train':
        trainer.train()
    elif trainer.args['mode'] == 'test':
        trainer.load(trainer.args['ckpt'])
        trainer.eval(trainer.start, mode = 'test')
