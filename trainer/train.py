import torch

from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import numpy as np
import os
from nets.resnet import ResNet
from metrics.accuracy import accuracy, balanced_accuracy

from dataset.custom_set import CustomSet

from configs.cfg_custom_set import cfg_custom_set
from configs.cfg_dataloader import cfg_dataloader
from configs.cfg_model import cfg_model
from configs.cfg_train import cfg_train

import time

class Trainer:
    def __init__(self, config_dataset, config_model, config_dataloader, config_train):
        self.config_dataset = config_dataset
        self.config_model = config_model
        self.config_dataloader = config_dataloader
        self.config_train = config_train
        self.max_bal_accuracy = 0
        self.need_save = False
        self.start_epoch = -1

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config_train.device == 'cuda' else 'cpu')

        print('Preparing data')
        self._get_dataset()
        self._get_dataloader()




        print('-'*100)
        print('Prepare model')
        if config_model.name_model == 'ResNet50':
            self.model = ResNet(config_model,config_model.nb_classes)

        self.model.to(self.device)

        decay = dict()
        no_decay = dict()
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                decay[name] = param
            else:
                no_decay[name] = param
        if self.config_train.loss in dir(nn):
            self.loss = getattr(nn, self.config_train.loss)(label_smoothing=cfg_train.label_smoothing)
        self.optimizer = getattr(optim, self.config_train.optimizer)(
            [{'params': decay.values(),
              'weight_decay': self.config_train.weight_decay,
              'lr': self.config_train.lr,
              'momentum': self.config_train.momentum
              },
            {'params': no_decay.values(),
             'weight_decay': 0,
             'lr': self.config_train.lr,
             'momentum': self.config_train.momentum
             }
             ]
            )
        if self.config_train.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config_train.nrof_epochs, eta_min=1e-7)
            self.warmup_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=100,
                                                      after_scheduler=self.lr_scheduler)

        print(f'Model: {self.config_model.name_model}\nLoss function: {self.config_train.loss}\nOptimizer: {self.config_train.optimizer}')

        if self.config_train.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config_train.log_path)
            # self.writer.add_hparams(self.config2dict(), {})

    def _get_dataset(self):
        self.dataset_train = CustomSet(cfg_custom_set.path, 'train',mixup_train=self.config_train.use_mixup_training)
        self.dataset_test = CustomSet(cfg_custom_set.path, 'valid')
        return self.dataset_train, self.dataset_test

    def _get_dataloader(self):
        if self.config_train.weight_sampling:
            nb_imgs_each_classes = np.array(self.dataset_train.get_nb_elem_in_each_class())
            class_weights = 1 - nb_imgs_each_classes / nb_imgs_each_classes.sum()
            lbls_sample = self.dataset_train.get_lbl_each_sample()
            weights = [class_weights[lbls_sample[i]] for i in range(nb_imgs_each_classes.sum())]
            sampler = torch.utils.data.WeightedRandomSampler(weights, int(nb_imgs_each_classes.sum()), replacement=False)
            shuffle = False
        else:
            sampler = None
            shuffle = self.config_dataloader.shuffle


        self.dataloader_train = DataLoader(self.dataset_train,
                                     batch_size=self.config_dataloader.batch_size,
                                     shuffle=shuffle,
                                     drop_last=self.config_dataloader.drop_last, num_workers=self.config_dataloader.num_workers,
                                           sampler=sampler)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=self.config_dataloader.batch_size, num_workers=self.config_dataloader.num_workers)

        return self.dataloader_train, self.dataloader_test




    def __train_epoch(self, epoch):
        for i, (batch_data, batch_label) in enumerate(self.dataloader_train):
            batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)

            pred = self.model(batch_data)

            loss = self.loss(pred, batch_label)

            if self.config_train.verbose or self.config_train.tensorboard:
                # acc = accuracy(pred.detach(), batch_label).item()
                # bal_acc = balanced_accuracy(pred.detach(), batch_label).item()
                if self.config_train.verbose:
                    print(f'epoch: [{epoch}/{self.config_train.nrof_epochs}], '
                      f'iter: [{i}/{len(self.dataloader_train)}], '
                      f'loss: {loss.item()}')
                      # f'accuracy: {acc}',
                      # f'balanced accuracy: {bal_acc}')


                else:
                    if i % 50 == 0:
                        print(f'epoch: [{epoch}/{self.config_train.nrof_epochs}], '
                              f'iter: [{i}/{len(self.dataloader_train)}], '
                              f'loss: {loss.item()}')
                              # f'accuracy: {acc}',
                              # f'balanced accuracy: {bal_acc}')

                if self.config_train.tensorboard:
                    self.writer.add_scalar('batch/loss', loss.item(), i + epoch * len(self.dataloader_train))
                    # self.writer.add_scalar('batch/accuracy', acc, i + epoch * len(self.dataloader_train))
                    # self.writer.add_scalar('batch/balanced_accuracy', bal_acc, i + epoch * len(self.dataloader_train))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self, dataloader, epoch=-1, type_data=None):
        total_loss = []
        total_accuracy = []
        total_balanced_accuracy = []
        start = time.time()
        with torch.no_grad():
            for batch_data, batch_label in dataloader:
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                predictions = self.model(batch_data)
                loss = self.loss(predictions, batch_label)
                total_accuracy.append(accuracy(predictions, batch_label).item())
                total_balanced_accuracy.append(balanced_accuracy(predictions, batch_label).item())
                total_loss.append(loss.item())

        total_accuracy = np.sum(total_accuracy)/len(dataloader)
        total_balanced_accuracy = np.sum(total_balanced_accuracy)/len(dataloader)
        total_loss = np.sum(total_loss)/len(dataloader)
        print(f'Eval time {time.time()-start} s.')
        if type_data == 'test' and total_balanced_accuracy > self.max_bal_accuracy:
            self.max_bal_accuracy = total_balanced_accuracy
            self.need_save = True

        print(f'epoch: {epoch}, total accuracy: {round(total_accuracy,3)*100}%, total loss: {total_loss}',
              f'balanced accuracy: {round(total_balanced_accuracy,3)*100}%')

        if self.config_train.tensorboard:
            self.writer.add_scalar(f'total/acc_{type_data}', total_accuracy, epoch)
            self.writer.add_scalar(f'total/loss_{type_data}', total_loss, epoch)
            self.writer.add_scalar(f'total/bal_accuracy_{type_data}', total_balanced_accuracy, epoch)

        return total_balanced_accuracy

    def fit(self):
        if self.config_train.load:
            self.load(self.config_train.type_checkpoint)

        for epoch in range(self.start_epoch+1, self.config_train.nrof_epochs):
            print(f'Epoch {epoch}')
            print('Train')
            import time
            s = time.time()
            self.model.train()
            self.__train_epoch(epoch)
            if self.config_train.scheduler:
                self.lr_scheduler.step(epoch)
                self.warmup_scheduler.step(epoch)
            self.model.eval()
            print(f'Iters in seconds {len(self.dataloader_train)/(time.time()-s)} ')
            if self.config_train.eval_on_train:
                print('Eval on train')
                bal_acc = self.eval(self.dataloader_train, epoch, 'train')

                # if self.config_train.scheduler and self.config_train.watch_on == 'train':
                #     self.lr_scheduler.step(bal_acc)
                #     self.writer.add_scalar(f'learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            if self.config_train.eval_on_test:
                print('Eval on test')
                bal_acc = self.eval(self.dataloader_test, epoch, 'test')

                # if self.config_train.scheduler and self.config_train.watch_on == 'test':
                #     self.lr_scheduler.step(bal_acc)
                #     self.writer.add_scalar(f'learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            if self.config_train.save_best_epoch and self.need_save:
                self.save(f'best_epoch', epoch)
                self.need_save = False
            if self.config_train.save_each_epoch:
                self.save(f'last_epoch', epoch)

    def save(self, name_save='fast_save', epoch=-1):
        print('Saving model...')
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'epoch': epoch,
                 'best_accuracy': self.max_bal_accuracy
                 }
        if not os.path.exists(self.config_train.save_folder):
            os.makedirs(self.config_train.save_folder)
        torch.save(state, os.path.join(self.config_train.save_folder, name_save))

    def load(self, name_save='best_epoch'):
        checkpoint = torch.load(os.path.join(self.config_train.load_folder, name_save))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        self.max_bal_accuracy = checkpoint['best_accuracy']
        print(f"Load from: {checkpoint['epoch']}")

    def config2dict(self):
        config = {'lr': self.config_train.lr,
                  'batch_size': self.config_dataloader.batch_size,
                  'augs': str(self.config_dataloader.transforms_train),
                  'optimizer': self.config_train.optimizer
                  }
        return config

    def fit_on_batch(self):
        batch_data, batch_label = next(iter(self.dataloader_train))
        for i in range(self.config_train.nrof_epochs):
            batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)

            pred = self.model(batch_data)
            loss = self.loss(pred, batch_label)

            if self.config_train.verbose or self.config_train.tensorboard:
                acc = accuracy(pred.detach(), batch_label).item()
                bal_acc = balanced_accuracy(pred.detach(), batch_label).item()

                if self.config_train.verbose:
                    print(f'epoch: [{i}/{self.config_train.nrof_epochs}], '
                      f'loss: {loss.item()}',
                      f'accuracy: {acc}',
                      f'balanced accuracy: {bal_acc}')

                else:
                    if i % 50 == 0:
                        print(f'epoch: [{i}/{self.config_train.nrof_epochs}], '
                              f'loss: {loss.item()}',
                              f'accuracy: {acc}',
                              f'balanced accuracy: {bal_acc}')

                if self.config_train.tensorboard:
                    self.writer.add_scalar('batch/loss', loss.item(), i)
                    self.writer.add_scalar('batch/accuracy', acc, i)
                    self.writer.add_scalar('batch/balanced_accuracy', bal_acc, i)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



if __name__ == '__main__':
    trainer = Trainer(config_dataset=cfg_custom_set,
                      config_model=cfg_model,
                      config_train=cfg_train,
                      config_dataloader=cfg_dataloader)
    # trainer.fit_on_batch()
    # trainer.fit()
    trainer.load()
    trainer.model.eval()
    trainer.eval(trainer.dataloader_test)
