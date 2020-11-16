# runner.py - specific training runner
# Author: hysxx
# Email: 1350460443@qq.com

import os
import copy
from glob import glob
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Runner():
    '''
        Args:
        params (Params): parameters, defined in train.py.
        net (Efficientnet): main model, defined in model.py.
        optim (torch.optim): optimizer of training
        torch_device (torch.cuda): used GPUs
        loss (torch.nn.modules.loss): loss function
        writer (SummaryWriter): log the details
        scheduler (torch.optim.lr_scheduler): scheduler to alter lr
    '''
    def __init__(self, params, net, optim, torch_device, loss, writer, scheduler, ckpt=None):
        self.params = params
        self.save_dir = params.save_dir
        self.result = os.path.join(self.save_dir, 'results.txt')  # training result will be log in this file
        self.writer = writer

        self.torch_device = torch_device

        self.net = net
        self.ema = copy.deepcopy(net.module).cpu()  # ema model
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema_decay = params.ema_decay

        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler

        self.start_epoch = 0
        self.best_metric = -1
        self.best_valid_loss = np.inf
        self.conf_thresh = params.conf_thresh

        if ckpt:
            self.load(ckpt)

    def save(self, epoch, filename="train"):

        torch.save({"model_level": self.params.model_level,
                    "start_epoch": epoch + 1,
                    "network": self.net.module.state_dict(),
                    "ema": self.ema.state_dict(),  # after training, either 'network' or 'ema' could be use
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric,
                    "best_valid_loss": self.best_valid_loss
                    }, self.save_dir + "/%s.pth" % (filename))
        print("Model saved %d epoch" % (epoch))

    # this function is unfinished
    def load(self, ckpt):
        self.ema.load_state_dict(ckpt['ema'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['start_epoch']
        self.best_metric = ckpt["best_metric"]
        # self.best_valid_loss = ckpt["best_valid_loss"]
        print("Model Loaded. Resuming training from epoch {}".format(self.start_epoch))

    # function to update ema model after every epoch
    def update_ema(self):
        with torch.no_grad():
            named_param = dict(self.net.module.named_parameters())
            for k, v in self.ema.named_parameters():
                param = named_param[k].detach().cpu()
                v.copy_(self.ema_decay * v + (1 - self.ema_decay) * param)

    def train(self, train_loader, val_loader=None):
        ni = len(train_loader)
        step = 0
        # self.net.train()
        for epoch in range(self.start_epoch, self.params.epoch):
            self.net.train()
            loss_ls = []  # loss list of one epoch
            pbar = tqdm(enumerate(train_loader), total=ni)
            for i, (input_, target_) in pbar:
                target_ = target_.to(self.torch_device, non_blocking=True)

                self.optim.zero_grad()
                out = self.net(input_)
                loss = self.loss(out, target_)
                loss_ls.append(loss.item())

                loss.backward()
                self.optim.step()
                self.update_ema()

                self.writer.add_scalars('BCEloss', {'train set': loss.item()}, step)

                # mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)
                pbar.set_description('Step{}. Epoch: {}/{}. Iteration: {}/{}. BCEloss: {:.5f}. '
                        .format(step, epoch, self.params.epoch-1, i,
                                ni-1, loss.item()))
                step += 1

            self.scheduler.step()
            if val_loader is not None:  # validation after every epoch
                loss, precision, recall, metric = self.valid(epoch, val_loader, step)
                print('valid loss: {}, precision: {}, recall: {}, metric: {}'.format(loss, precision, recall, metric))
            else:
                raise RuntimeError('val_loader not existed!')

            self.writer.add_scalar('lr', self.optim.state_dict()['param_groups'][0]['lr'], epoch)

            with open(self.result, 'a') as f:  # save results of this epoch to file
                f.write('Epoch: {}/{}  Train loss: {:.5f}  Val loss: {: .5f}  Presicion: {: .5f}  Recall: {: .5f}  Metric: {: .5f}'
                        .format(epoch, self.params.epoch-1, np.mean(loss_ls), loss, precision, recall, metric) + '\n')


    # calculate mean loss, precision and recall
    def _get_acc(self, loader, singlecls=False):
        
        loss_ls = []
        preds = []
        targets = []

        tp_ = 0  # true positive
        fp_ = 0  # false positive
        tn_ = 0  # true negative
        fn_ = 0  # false negative

        with torch.no_grad():
            
            for input_, target_ in loader:
                target_ = target_.to(self.torch_device, non_blocking=True)
                outs = self.net(input_)
                loss_ls.append(self.loss(outs, target_).item())

                if self.params.multilabels:
                    outs = torch.sigmoid(outs).cpu().numpy()
                else:
                    outs = F.softmax(outs).cpu().numpy()

                for i, out in enumerate(outs):
                    pred = [1 if x else 0 for x in out > self.conf_thresh]

                    if singlecls:
                        preds.append(pred)
                        targets.append(target_[i].cpu().numpy())

                    for x in range(len(pred)):
                        if pred[x] == 1 and target_[i, x] == 1:
                            tp_ += 1
                        elif pred[x] == 1 and target_[i, x] == 0:
                            fp_ += 1
                        elif pred[x] == 0 and target_[i, x] == 0:
                            tn_ += 1
                        elif pred[x] == 0 and target_[i, x] == 1:
                            fn_ += 1
                        else:
                            raise ValueError('preds is not binary!')

        if singlecls:
            result_sc = []
            preds = np.array(preds)
            targets = np.array(targets)
            assert preds.shape == targets.shape

            for j in range(preds.shape[1]):
                tp, fp, tn, fn = 0, 0, 0, 0

                for i in range(preds.shape[0]):
                    if preds[i, j] == 1 and targets[i, j] == 1:
                        tp += 1
                    elif preds[i, j] == 1 and targets[i, j] == 0:
                        fp += 1
                    elif preds[i, j] == 0 and targets[i, j] == 0:
                        tn += 1
                    elif preds[i, j] == 0 and targets[i, j] == 1:
                        fn += 1
                    else:
                        raise ValueError('preds is not binary!')
                
                p = tp/(tp+fp) if tp+fp!=0 else 0
                r = tp/(tp+fn) if tp+fn!=0 else 0
                f1 = 2*p*r / (p+r) if p+r!=0 else 0
                result_sc.append([p, r, f1])
                    
                    
            for i, (p, r) in enumerate(result_sc):
                print('{}st class: precision {:.3f} recall {:.3f} f1 {:.3f}'.format(i, p, r, f1))

        return np.mean(loss_ls), tp_/(tp_+fp_), tp_/(tp_+fn_)  # mean loss, precision, recall


    # only calculate mean loss
    def _get_loss(self, loader):
        with torch.no_grad():
            loss_ls = []
            for input_, target_ in loader:
                target_ = target_.to(self.torch_device, non_blocking=True)
                out = self.net(input_)
                loss = self.loss(out, target_).item()
                loss_ls.append(loss)
        return np.mean(loss_ls)

    # calculate metric(s)
    def _metric(self, precision, recall):
        metric = 2 * (precision * recall) / (precision + recall)    # F1 value
        return metric

    # validation function
    def valid(self, epoch, val_loader, step):
        
        print('validating...')
        self.net.eval()
        loss, precision, recall = self._get_acc(val_loader, singlecls=True)

        metric = self._metric(precision, recall)

        self.writer.add_scalars('BCEloss', {'val set': loss}, step)
        self.writer.add_scalar('Precision', precision, epoch)
        self.writer.add_scalar('Recall', recall, epoch)
        self.writer.add_scalar('Metric', metric, epoch)

        if metric > self.best_metric:  # save the model if metric is better than before
            self.best_metric = metric
            self.save(epoch, "ckpt_%d_%.4f" % (
                epoch, metric))
                
        return loss, precision, recall, metric
