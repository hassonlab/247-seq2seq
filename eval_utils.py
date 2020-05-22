import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def best_threshold(X, Y, T, best_x=0., best_y=1.):
    '''
    ### Choose point of minimum distance to an ideal point,
    ### (For ROC: (0,1); for PR: (1,1)).
    '''
    min_d, min_i = np.inf, 0
    for i, (x, y) in enumerate(zip(X, Y)):
        d = np.sqrt((best_x - x)**2 + (best_y - y)**2)
        if d < min_d:
            min_d, min_i = d, i
    return X[min_i], Y[min_i], T[min_i]


def evaluate_roc(predictions,
                 labels,
                 i2w,
                 train_freqs,
                 save_dir,
                 do_plot,
                 given_thresholds=None,
                 title='',
                 suffix='',
                 min_train=10,
                 tokens_to_remove=[]):
    '''### Evaluate ROC performance of the model
    ### (predictions, labels of shape (n_examples, n_classes))
    '''
    assert (predictions.shape == labels.shape)
    lines, scores, word_freqs = [], [], []
    n_examples, n_classes = predictions.shape
    thresholds = np.full(n_classes, np.nan)
    rocs, fprs, tprs = {}, [], []

    # Create directory for plots if required
    if do_plot:
        roc_dir = save_dir + 'rocs/'
        if not os.path.isdir(roc_dir):
            os.mkdir(roc_dir)

    # Go over each class and compute AUC
    for i in range(n_classes):
        if i2w[i] in tokens_to_remove:
            continue
        train_count = train_freqs[i]
        n_true = np.count_nonzero(labels[:, i])
        if train_count < 1 or n_true == 0:
            continue
        word = i2w[i]
        probs = predictions[:, i]
        c_labels = labels[:, i]
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
        lines.append('%s\t%3d\t%3d\t%.3f\t%d\t%d\t%d\t%d\n' %
                     (word, n_true, train_count, score, tp, fp, fn, tn))
        if do_plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].plot(fpr, tpr, color='darkorange', lw=2, marker='.')
            axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0].plot(x, y, marker='o', color='blue')
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.05])
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            h1 = probs[c_labels == 1].reshape(-1)
            h2 = probs[c_labels == 0].reshape(-1)
            axes[1].hist(h2,
                         bins=20,
                         color='orange',
                         alpha=0.5,
                         label='Neg. Examples')
            # axes[1].twinx().hist(h1, bins=50, alpha=0.5, label='Pos. Examples')
            axes[1].hist(h1, bins=50, alpha=0.5, label='Pos. Examples')
            axes[1].axvline(threshold, color='k')
            axes[1].set_xlabel('Activation')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].set_title('%d TP | %d FP | %d FN | %d TN' %
                              (tp, fp, fn, tn))
            fig.suptitle('ROC Curve | %s | AUC = %.3f | N = %d' %
                         (word, score, n_true))
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
    _, ax = plt.subplots(1, 1)
    ax.scatter(word_freqs, scores, marker='.')
    ax.set_xlabel('# examples')
    ax.set_ylabel('AUC')
    ax.set_title('%s | avg: %.3f | N = %d' %
                 (title, weighted_avg, scores.size))
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    ax.grid()
    plt.savefig(save_dir + 'roc-auc-examples.png', bbox_inches='tight')

    _, ax = plt.subplots(1, 1)
    ax.hist(scores, bins=20)
    ax.set_xlabel('AUC')
    ax.set_ylabel('# labels')
    ax.set_title('%s | avg: %.3f | N = %d' %
                 (title, weighted_avg, scores.size))
    ax.set_xticks(np.arange(0., 1., 0.1))
    plt.savefig(save_dir + 'roc-auc.png', bbox_inches='tight')

    _, ax = plt.subplots(1, 1)
    for fpr, tpr in zip(fprs, tprs):
        ax.plot(fpr, tpr, lw=1)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('%s | avg: %.3f | N = %d' %
                 (title, weighted_avg, scores.size))
    plt.savefig(save_dir + 'roc-auc-all.png', bbox_inches='tight')

    return {
        'rocauc_avg': avg_auc,
        'rocauc_stddev': scores.std(),
        'rocauc_w_avg': weighted_avg,
        'rocauc_n': scores.size,
        'rocs': rocs
    }


def evaluate_topk(predictions,
                  labels,
                  i2w,
                  train_freqs,
                  save_dir,
                  min_train=10,
                  prefix='',
                  suffix='',
                  tokens_to_remove=[]):
    '''
    ### Evaluate top-k performance of the model. (assumes activations can be
    ### interpreted as probabilities).
    ### (predictions, labels of shape (n_examples, n_classes))
    '''
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
    freqs = np.array([freqs[i] for i, _ in train_freqs.most_common()])
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
        prefix + 'top1_chance': chances[0],
        prefix + 'top5_chance': chances[4],
        prefix + 'top10_chance': chances[9],
        prefix + 'top1_above': (top1 - chances[0]) / chances[0],
        prefix + 'top5_above': (top5 - chances[4]) / chances[4],
        prefix + 'top10_above': (top10 - chances[9]) / chances[9],
        prefix + 'top1_n_uniq_correct': len(top1_uw),
        prefix + 'top5_n_uniq_correct': len(top5_uw),
        prefix + 'top10_n_uniq_correct': len(top10_uw),
        prefix + 'word_accuracies': accs
    }
