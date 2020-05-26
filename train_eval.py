import math
import re
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CLIP_NORM = 1.0
REGEX = re.compile('[^a-zA-Z]')


class NoamOpt:
    "Optimizer wrapper implementing learning scheme"

    def __init__(self, d_model, prefactor, warmup, optimizer):
        self.d_model = d_model
        self.optimizer = optimizer
        self.warmup = warmup
        self.prefactor = prefactor
        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement learning rate warmup scheme"
        if step is None:
            step = self._step
        return self.prefactor * (self.d_model**(-0.5) *
                                 min(step**(-0.5), step * self.warmup**(-1.5)))


class LabelSmoothing(nn.Module):
    "Implements label smoothing on a multiclass target."

    def __init__(self, criterion, size, pad_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = criterion
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert (x.size(1) == self.size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.sum() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None, scheduler=None):
        self.criterion = criterion
        self.opt = opt
        self.scheduler = scheduler

    def __call__(self, x, y, val=False):
        loss = self.criterion(x.view(-1, x.size(-1)),
                              y.view(-1))
        if not val:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return loss.data.item()


def train(data_iter,
          model,
          criterion,
          devices,
          device,
          opt,
          scheduler=None,
          seq2seq=False,
          pad_idx=-1):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_acc = 0.
    count, batch_count = 0, 0
    for i, batch in enumerate(data_iter):
        # Prevent gradient accumulation
        model.zero_grad()
        src = batch[0].to(device)
        trg = batch[1].long().to(device)
        if seq2seq:
            trg_y = batch[2].long().to(device)
            trg_pos_mask, trg_pad_mask = batch[3].to(device), batch[4].to(
                device)
            # Perform loss computation during forward pass for parallelism
            out, trg_y, loss = model.forward(src, trg, trg_pos_mask,
                                             trg_pad_mask, trg_y, criterion)
            total_loss += loss.data.item()
            """ idx = (trg_y != pad_idx).nonzero(as_tuple=True)
            out = out[idx]
            trg_y = trg_y[idx]
            out = torch.argmax(out, dim=1)
            total_acc += float((out == trg_y).sum()) """
            opt.step()
            if scheduler is not None:
                scheduler.step()
            # total_loss += loss.data.item()
            # out = out[idx]
            # trg_y = trg_y[idx]
            # out = torch.argmax(out, dim=1)
            # total_acc += float((out == trg_y).sum())
            # print("hereo")
            # sys.stdout.flush()
        else:
            out = model.forward(src)
            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            loss.backward()
            if opt is not None:
                opt.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.data.item()
            total_acc += float((torch.argmax(out, dim=1) == trg).sum())
            # count += int(out.size(0))
        # Prevent gradient blowup
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        count += int(out.size(0))
        batch_count += 1
    total_loss /= batch_count
    # total_acc /= count
    elapsed = (time.time() - start_time) * 1000. / batch_count
    perplexity = float('inf')
    perplexity = math.exp(total_loss)
    print(
        'loss {:5.3f} | perplexity {:3.2f} | ms/batch {:5.2f}'
        .format(total_loss, perplexity, elapsed),
        end='')
    return total_loss, total_acc


def valid(data_iter,
          model,
          criterion,
          device,
          temperature=1.0,
          n_samples=10,
          seq2seq=False,
          pad_idx=-1):
    model.eval()
    total_loss = 0.
    total_acc = 0.
    total_sample_rank_acc = 0.
    batch_count, count = 0, 0
    for i, batch in enumerate(data_iter):
        src = batch[0].to(device)
        trg = batch[1].long().to(device)
        if seq2seq:
            trg_y = batch[2].long().to(device)
            trg_pos_mask, trg_pad_mask = batch[3].to(device), batch[4].to(
                device)
            out, trg_y, loss = model.forward(src, trg, trg_pos_mask,
                                             trg_pad_mask, trg_y, criterion)
            total_loss += loss.data.item()
            """ idx = (trg_y != pad_idx).nonzero(as_tuple=True)
            out = out[idx]
            trg_y = trg_y[idx]
            out_top1 = torch.argmax(out, dim=1)
            total_acc += float((out_top1 == trg_y).sum())
            out = F.softmax(out / temperature, dim=1)
            samples = torch.multinomial(out, n_samples)
            pred = torch.zeros(samples.size(0)).to(device)
            for j in range(len(pred)):
                pred[j] = samples[j, torch.argmax(out[j, samples[j]])]
            total_sample_rank_acc += float((pred == trg_y).sum()) """
        else:
            out = model.forward(src)
            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            total_loss += loss.data.item()
            total_acc += float((torch.argmax(out, dim=1) == trg).sum())
            out = F.softmax(out / temperature, dim=1)
            samples = torch.multinomial(out, n_samples)
            pred = torch.zeros(samples.size(0)).cuda()
            for j in range(len(pred)):
                pred[j] = samples[j, torch.argmax(out[j, samples[j]])]
            total_sample_rank_acc += float((pred == trg).sum())
        count += int(out.size(0))
        batch_count += 1
    total_loss /= batch_count
    # total_acc /= count
    # total_sample_rank_acc /= count
    perplexity = float('inf')
    perplexity = math.exp(total_loss)
    print(
        'loss {:5.3f} | perplexity {:3.2f}'
        .format(total_loss, perplexity))
    return total_loss, total_acc


def plot_training(history, save_dir, title='', val=True):
    '''Plot train/val loss and accuracy and save figures'''
    plt.plot(history["train_loss"])
    if val:
        plt.plot(history["valid_loss"])
    plt.title('Model loss: %s' % title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_dir + 'loss.png')
    plt.clf()
    plt.plot(history["train_acc"])
    if val:
        plt.plot(history["valid_acc"])
    plt.title('Model accuracy: %s' % title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_dir + 'accuracy.png')
