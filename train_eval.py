################################################################################
#
# Brain2En > Training and Evaluation Utilities
#
################################################################################

import math
import os
import re
import sys
import time
### Libraries
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.autograd import Variable

################################################################################
#
# Optimization Classes and Methods
#
################################################################################


CLIP_NORM = 1.0
REGEX = re.compile('[^a-zA-Z]')

### NOAM Optimizer
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

    def rate(self, step = None):
        "Implement learning rate warmup scheme"
        if step is None:
            step = self._step
        return self.prefactor * (self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

### Regularization by Label Smoothing
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
        assert(x.size(1) == self.size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1,
            target.data.unsqueeze(1).long(),
            self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.sum() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

### Single GPU Loss Computation
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, criterion, opt=None, scheduler=None):
        self.criterion = criterion
        self.opt = opt
        self.scheduler = scheduler

    def __call__(self, x, y, val=False):
        loss = self.criterion(x.view(-1, x.size(-1)),\
                              y.view(-1))
        if not val:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return loss.data.item()

### Multi GPU Loss Computation
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, criterion, devices, opt=None, scheduler=None, chunk_size=5):
        # Send out to different gpus.
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=devices)
        self.opt = opt
        self.scheduler = scheduler
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, x, y, val=False):
        total = 0.0
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data,
                                    requires_grad=self.opt is not None)]
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i+chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            l = l.sum()[0]
            total += l.data[0]

            # Backprop loss to output of transformer
            if not val and self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())
        if not val:
            if self.opt is not None:
                out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
                o1 = out
                o2 = nn.parallel.gather(out_grad,
                                        target_device=self.devices[0])
                o1.backward(gradient=o2)
                self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return total


################################################################################
#
# Training Methods
#
################################################################################


### Training loop
def train(data_iter, model, criterion, devices, device, opt,
          scheduler=None, seq2seq=False, pad_idx=-1):
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
            # trg_y = batch[2].long().to(src.device)
            # trg_pos_mask, trg_pad_mask = batch[3].to(src.device), batch[4].to(src.device)
            # out, trg_y = model.forward(src, trg, trg_y, trg_pos_mask, trg_pad_mask)
            # # Fix asymmetrical load on single GPU by computing loss in parallel
            # out_scatter = nn.parallel.scatter(out, target_gpus=devices)
            # out_grad = [[] for _ in out_scatter]
            # trg_y_scatter = nn.parallel.scatter(trg_y, target_gpus=devices)
            # chunk_size = int(math.ceil(out_scatter[0].size(0)/len(devices)))
            # for i in range(0, out_scatter[0].size(0), chunk_size):
            #     out_scatter_chunk = [Variable(o[i:end], requires_grad=True)\
            #                          for o in out_scatter if (end := min(i+chunk_size,len(o))) > i]
            #     trg_y_scatter_chunk = [t[i:end] for t in trg_y_scatter\
            #                            if (end := min(i+chunk_size, len(t))) > i]
            #     idx_scatter = [(t != pad_idx).nonzero(as_tuple=True) for t in trg_y_scatter_chunk]
            #     y = [(o.contiguous().view(-1, o.size(-1)), t.contiguous().view(-1)) for o, t in zip(out_scatter_chunk, trg_y_scatter_chunk)]
            #     print(len(criterion), len(y))
            #     print([(o.size(), t.size()) for o, t in y])
            #     loss = nn.parallel.parallel_apply(criterion[:len(y)], y)
            #     loss = [l.unsqueeze(-1) for l in loss]
            #     num_dev = len(loss)
            #     # print(num_dev)
            #     loss = nn.parallel.gather(loss, target_device=devices[0])
            #     loss = loss.sum()
            #     total_loss += float(loss.item() / num_dev)
            #     loss.backward()
            #     for j in range(num_dev):
            #         out_grad[j].append(out_scatter_chunk[j].grad.data.clone())
            #     out_scatter_chunk = [torch.argmax(o[idx], dim=1) for idx, o in zip(idx_scatter, out_scatter_chunk)]
            #     trg_y_scatter_chunk = [t[idx] for idx, t in zip(idx_scatter, trg_y_scatter_chunk)]
            #     total_acc += sum([float((o == t).sum()) for o, t in zip(out_scatter_chunk, trg_y_scatter_chunk)])
            #     count += sum(int(o.size(0)) for o in out_scatter_chunk)
            #     # del out_scatter_chunk, trg_y_scatter_chunk, idx_scatter, y, loss
            #     print(total_loss, total_acc / count)
            #     sys.stdout.flush()
            # print("Here now")
            # out_grad = [Variable(torch.cat(og, dim=0)) for og in out_grad]
            # print(out.size(), trg_y.size())
            # print([o.size() for o in out_grad])
            # sys.stdout.flush()
            # out.backward(gradient=nn.parallel.gather(out_grad,
            #                                          target_device=devices[0]))
            # del src, trg, trg_y, trg_pos_mask, trg_pad_mask, out, out_grad
            # print("brah")
            # loss = criterion(out.view(-1, out.size(-1)), trg_y.view(-1))
            # loss.backward()
            trg_y = batch[2].long().to(device)
            trg_pos_mask, trg_pad_mask = batch[3].to(device), batch[4].to(device)
            # Perform loss computation during forward pass for parallelism
            out, trg_y, loss = model.forward(src, trg, trg_pos_mask, trg_pad_mask, trg_y, criterion)
            idx = (trg_y != pad_idx).nonzero(as_tuple=True)
            total_loss += loss.data.item()
            out = out[idx]
            trg_y = trg_y[idx]
            out = torch.argmax(out, dim=1)
            total_acc += float((out == trg_y).sum())
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
    total_acc /= count
    elapsed = (time.time() - start_time) * 1000. / batch_count
    perplexity = float('inf')
    try:
        perplexity = math.exp(total_loss)
    except:
        pass
    print('loss {:5.3f} | accuracy {:5.3f} | perplexity {:3.2f} | ms/batch {:5.2f}'.format(total_loss, total_acc, perplexity, elapsed), end='')
    return total_loss, total_acc

### Validation loop
def valid(data_iter, model, criterion, device,
          temperature=1.0, n_samples=10, seq2seq=False, pad_idx=-1):
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
            trg_pos_mask, trg_pad_mask = batch[3].to(device), batch[4].to(device)
            out, trg_y, loss = model.forward(src, trg, trg_pos_mask, trg_pad_mask, trg_y, criterion)
            idx = (trg_y != pad_idx).nonzero(as_tuple=True)
            total_loss += loss.data.item()
            out = out[idx]
            trg_y = trg_y[idx]
            out_top1 = torch.argmax(out, dim=1)
            total_acc += float((out_top1 == trg_y).sum())
            out = F.softmax(out/temperature, dim=1)
            samples = torch.multinomial(out, n_samples)
            pred = torch.zeros(samples.size(0)).to(device)
            for j in range(len(pred)):
                pred[j] = samples[j,torch.argmax(out[j,samples[j]])]
            total_sample_rank_acc += float((pred == trg_y).sum())
        else:
            out = model.forward(src)
            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            total_loss += loss.data.item()
            total_acc += float((torch.argmax(out, dim=1) == trg).sum())
            out = F.softmax(out/temperature, dim=1)
            samples = torch.multinomial(out, n_samples)
            pred = torch.zeros(samples.size(0)).cuda()
            for j in range(len(pred)):
                pred[j] = samples[j,torch.argmax(out[j,samples[j]])]
            total_sample_rank_acc += float((pred == trg).sum())
        count += int(out.size(0))
        batch_count += 1
    total_loss /= batch_count
    total_acc /= count
    total_sample_rank_acc /= count
    perplexity = float('inf')
    try:
        perplexity = math.exp(total_loss)
    except:
        pass
    print('loss {:5.3f} | accuracy {:5.3f} | sample-rank acc {:5.3f} | perplexity {:3.2f}'.format(total_loss, total_acc, total_sample_rank_acc, perplexity))
    return total_loss, total_acc

### Plot train/val loss and accuracy and save figures
def plot_training(history, save_dir, title='', val=True):
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


################################################################################
#
# Evaluation Methods
#
################################################################################


### Choose point of minimum distance to an ideal point,
### (For ROC: (0,1); for PR: (1,1)).
def best_threshold(X, Y, T, best_x=0., best_y=1.):
    min_d, min_i = np.inf, 0
    for i, (x, y) in enumerate(zip(X, Y)):
        d = np.sqrt((best_x-x)**2 + (best_y-y)**2)
        if d < min_d:
            min_d, min_i = d, i
    return X[min_i], Y[min_i], T[min_i]

### Evaluate ROC performance of the model
### (predictions, labels of shape (n_examples, n_classes))
def evaluate_roc(predictions, labels, i2w, train_freqs, save_dir, do_plot,
                 given_thresholds=None, title='', suffix='', min_train=10,
                 tokens_to_remove=[]):
    assert(predictions.shape == labels.shape)
    lines, scores, word_freqs = [], [], []
    n_examples, n_classes = predictions.shape
    thresholds = np.full(n_classes, np.nan)
    rocs, fprs, tprs = {}, [], []

    # Create directory for plots if required
    if do_plot:
        roc_dir = save_dir + 'rocs/'
        if not os.path.isdir(roc_dir): os.mkdir(roc_dir)

    # Go over each class and compute AUC
    for i in range(n_classes):
        if i2w[i] in tokens_to_remove:
            continue
        train_count = train_freqs[i]
        n_true = np.count_nonzero(labels[:,i])
        if train_count < 1 or n_true == 0: continue
        word = i2w[i]
        probs = predictions[:,i]
        c_labels = labels[:,i]
        fpr, tpr, thresh = roc_curve(c_labels, probs)
        if given_thresholds is None:
            x, y, threshold = best_threshold(fpr, tpr, thresh)
        else:
            x, y, threshold = 0, 0, given_thresholds[i]
        thresholds[i] = threshold
        score = auc(fpr, tpr)
        scores.append(score)
        word_freqs.append(train_count)
        rocs[word] = score
        fprs.append(fpr)
        tprs.append(tpr)
        y_pred = probs >= threshold
        tn, fp, fn, tp = confusion_matrix(c_labels, y_pred).ravel()
        lines.append('%s\t%3d\t%3d\t%.3f\t%d\t%d\t%d\t%d\n' \
                % (word, n_true, train_count, score, tp, fp, fn, tn))
        if do_plot:
            fig, axes = plt.subplots(1,2, figsize=(16,6))
            axes[0].plot(fpr, tpr, color='darkorange', lw=2, marker='.')
            axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0].plot(x, y, marker='o', color='blue')
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.05])
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            h1 = probs[c_labels == 1].reshape(-1)
            h2 = probs[c_labels == 0].reshape(-1)
            axes[1].hist(h2, bins=20, color='orange',
                             alpha=0.5, label='Neg. Examples')
            #axes[1].twinx().hist(h1, bins=50, alpha=0.5, label='Pos. Examples')
            axes[1].hist(h1, bins=50, alpha=0.5, label='Pos. Examples')
            axes[1].axvline(threshold, color='k')
            axes[1].set_xlabel('Activation')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].set_title('%d TP | %d FP | %d FN | %d TN'\
                              % (tp, fp, fn, tn))
            fig.suptitle('ROC Curve | %s | AUC = %.3f | N = %d'\
                         % (word, score, n_true))
            plt.savefig(roc_dir + '%s.png' % word)
            fig.clear()
            plt.close(fig)

    # Compute statistics
    scores, word_freqs = np.array(scores), np.array(word_freqs)
    normed_freqs = word_freqs / word_freqs.sum()
    avg_auc = scores.mean()
    weighted_avg = (scores * normed_freqs).sum()
    print('Avg AUC: %d\t%.6f' % (scores.size, avg_auc))
    print('Weighted Avg AUC: %d\t%.6f' % (scores.size, weighted_avg))

    # Write to file
    with open(save_dir + 'aucs%s.txt' % suffix, 'w') as fout:
        for line in lines:
            fout.write(line)

    # Plot histogram and AUC as a function of num of examples
    _, ax = plt.subplots(1,1)
    ax.scatter(word_freqs, scores, marker='.')
    ax.set_xlabel('# examples')
    ax.set_ylabel('AUC')
    ax.set_title('%s | avg: %.3f | N = %d' % (title, weighted_avg, scores.size))
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    ax.grid()
    plt.savefig(save_dir + 'roc-auc-examples.png', bbox_inches='tight')

    _, ax = plt.subplots(1,1)
    ax.hist(scores, bins=20)
    ax.set_xlabel('AUC')
    ax.set_ylabel('# labels')
    ax.set_title('%s | avg: %.3f | N = %d' % (title, weighted_avg, scores.size))
    ax.set_xticks(np.arange(0., 1., 0.1))
    plt.savefig(save_dir + 'roc-auc.png', bbox_inches='tight')

    _, ax = plt.subplots(1,1)
    for fpr, tpr in zip(fprs, tprs):
        ax.plot(fpr, tpr, lw=1)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('%s | avg: %.3f | N = %d' % (title, weighted_avg, scores.size))
    plt.savefig(save_dir + 'roc-auc-all.png', bbox_inches='tight')

    return {
        'rocauc_avg': avg_auc,
        'rocauc_stddev': scores.std(),
        'rocauc_w_avg': weighted_avg,
        'rocauc_n': scores.size,
        'rocs': rocs
    }

### Evaluate top-k performance of the model. (assumes activations can be
### interpreted as probabilities).
### (predictions, labels of shape (n_examples, n_classes))
def evaluate_topk(predictions, labels, i2w, train_freqs, save_dir,
                  min_train=10, prefix='', suffix='', tokens_to_remove=[]):
    ranks = []
    n_examples, n_classes = predictions.shape
    fid = open(save_dir + 'guesses%s.csv' % suffix, 'w')
    top1_uw, top5_uw, top10_uw = set(), set(), set()
    accs, sizes = {}, {}
    total_freqs = float(sum(train_freqs.values()))

    # Go through each example and calculate its rank and top-k
    for i in range(n_examples):
        y_true_idx = labels[i]

        if train_freqs[y_true_idx] < 1:
            continue

        word = i2w[y_true_idx]
        if word in tokens_to_remove:
            continue

        # Get example predictions
        ex_preds = np.argsort(predictions[i])[::-1]
        rank = np.where(y_true_idx == ex_preds)[0][0]
        ranks.append(rank)

        fid.write('%s,%d,' % (word, rank))
        fid.write(','.join(i2w[j] for j in ex_preds[:10]))
        fid.write('\n')

        if rank == 0:
            top1_uw.add(ex_preds[0])
        elif rank < 5:
            top5_uw.update(ex_preds[:5])
        elif rank < 10:
            top10_uw.update(ex_preds[:10])

        if word not in accs:
            accs[word] = float(rank == 0)
            sizes[y_true_idx] = 1.
        else:
            accs[word] += float(rank == 0)
            sizes[y_true_idx] += 1.
    for idx in sizes:
        word = i2w[idx]
        chance_acc = float(train_freqs[idx]) / total_freqs * 100.
        if sizes[idx] > 0:
            rounded_acc = round(accs[word] / sizes[idx] * 100, 3)
            accs[word] = (rounded_acc, chance_acc, rounded_acc - chance_acc)
        else:
            accs[word] = (0., chance_acc, -chance_acc)
    accs = sorted(accs.items(), key=lambda x: -x[1][2])

    fid.close()
    print('Top1 #Unique:', len(top1_uw))
    print('Top5 #Unique:', len(top5_uw))
    print('Top10 #Unique:', len(top10_uw))

    n_examples = len(ranks)
    ranks = np.array(ranks)
    top1 = sum(ranks == 0) / (1e-12 + len(ranks)) * 100
    top5 = sum(ranks < 5) / (1e-12 + len(ranks)) * 100
    top10 = sum(ranks < 10) / (1e-12 + len(ranks)) * 100

    # Calculate chance levels based on training word frequencies
    freqs = Counter(labels)
    freqs = np.array([freqs[i] for i,_ in train_freqs.most_common()])
    freqs = freqs[freqs > 0]
    chances = (freqs / freqs.sum()).cumsum() * 100

    # Print and write to file
    if suffix is not None:
        with open(save_dir + 'topk%s.txt' % suffix, 'w') as fout:
            line = 'n_classes: %d\nn_examples: %d' % (n_classes, n_examples)
            print(line)
            fout.write(line + '\n')
            line = 'Top-1\t%.4f %% (%.2f %%)' % (top1, chances[0])
            print(line)
            fout.write(line + '\n')
            line = 'Top-5\t%.4f %% (%.2f %%)' % (top5, chances[4])
            print(line)
            fout.write(line + '\n')
            line = 'Top-10\t%.4f %% (%.2f %%)' % (top10, chances[9])
            print(line)
            fout.write(line + '\n')

    return {
        prefix + 'top1': top1,
        prefix + 'top5': top5,
        prefix + 'top10': top10,
        prefix + 'top1_chance':  chances[0],
        prefix + 'top5_chance':  chances[4],
        prefix + 'top10_chance': chances[9],
        prefix + 'top1_above':  (top1 - chances[0]) / chances[0],
        prefix + 'top5_above':  (top5 - chances[4]) / chances[4],
        prefix + 'top10_above': (top10 - chances[9]) / chances[9],
        prefix + 'top1_n_uniq_correct': len(top1_uw),
        prefix + 'top5_n_uniq_correct': len(top5_uw),
        prefix + 'top10_n_uniq_correct': len(top10_uw),
        prefix + 'word_accuracies': accs
    }
