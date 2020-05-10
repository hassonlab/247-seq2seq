import json
import math
import os
import random
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from transformers import AdamW

from arg_parser import arg_parser
from config import build_config
from data_util import Brain2enDataset, MyCollator
from build_matrices import (build_design_matrices_classification,
                            build_design_matrices_seq2seq)
from models import MeNTAL, PITOM, ConvNet10, MeNTALmini
from train_eval import evaluate_roc, evaluate_topk, plot_training, train, valid
from vocab_builder import get_vocab, get_sp_vocab

# from train_eval import *

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%A %d/%m/%Y %H:%M:%S")
print("Start Time: ", dt_string)
results_str = now.strftime("%Y-%m-%d-%H:%M")

args = arg_parser()
CONFIG = build_config(args, results_str)

# Model objectives
MODEL_OBJ = {
    "ConvNet10": "classifier",
    "PITOM": "classifier",
    "MeNTALmini": "classifier",
    "MeNTAL": "seq2seq"
}

# GPUs
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.gpus = min(args.gpus, torch.cuda.device_count())

# Fix random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

args.model = args.model.split("_")[0]
classify = False if (args.model in MODEL_OBJ
                     and MODEL_OBJ[args.model] == "seq2seq") else True

CONV_DIRS = CONFIG["CONV_DIRS"]
SAVE_DIR = CONFIG["SAVE_DIR"]
TRAIN_CONV = CONFIG["TRAIN_CONV"]
VALID_CONV = CONFIG["VALID_CONV"]

# Load train and validation datasets
# (if model is seq2seq, using speaker switching for sentence cutoff,
# and custom batching)
if classify:
    print("Building vocabulary")
    word2freq, vocab, n_classes, w2i, i2w = get_vocab(CONFIG)

    # Save word counter
    print("Saving word counter")
    with open("%sword2freq.json" % SAVE_DIR, "w") as fp:
        json.dump(word2freq, fp, indent=4)
    sys.stdout.flush()

    print("Loading training data")
    x_train, y_train = build_design_matrices_classification(
        CONFIG, w2i, TRAIN_CONV, delimiter=" ", aug_shift_ms=[-1000])
    sys.stdout.flush()
    print("Loading validation data")
    x_valid, y_valid = build_design_matrices_classification(CONFIG,
                                                            w2i,
                                                            VALID_CONV,
                                                            delimiter=" ",
                                                            aug_shift_ms=[])
    sys.stdout.flush()
    if args.model == "ConvNet10":
        x_train = x_train[:, np.newaxis, ...]
        x_valid = x_valid[:, np.newaxis, ...]
    # Shuffle labels if required
    if args.shuffle:
        print("Shuffling labels")
        np.random.shuffle(y_train)
        np.random.shuffle(y_valid)
    x_train = torch.from_numpy(x_train).float()
    print("Shape of training signals: ", x_train.size())
    y_train = torch.from_numpy(y_train)
    train_ds = data.TensorDataset(x_train, y_train)
    x_valid = torch.from_numpy(x_valid).float()
    print("Shape of validation signals: ", x_valid.size())
    y_valid = torch.from_numpy(y_valid)
    valid_ds = data.TensorDataset(x_valid, y_valid)
    # Create dataset and data generators
    print("Creating dataset and generators")
    sys.stdout.flush()
    train_dl = data.DataLoader(train_ds,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=CONFIG["num_cpus"])
    valid_dl = data.DataLoader(valid_ds,
                               batch_size=args.batch_size,
                               num_workers=CONFIG["num_cpus"])
else:
    print("Building vocabulary")
    vocab = get_sp_vocab(CONFIG, algo='unigram', vocab_size=500)
    # print([(i, vocab.IdToPiece(i)) for i in range(len(vocab))])

    print("Loading training data")
    x_train, y_train = build_design_matrices_seq2seq(
        CONFIG, vocab, TRAIN_CONV, delimiter=" ", aug_shift_ms=[-1000, -500])
    sys.stdout.flush()
    print("Loading validation data")
    x_valid, y_valid = build_design_matrices_seq2seq(CONFIG,
                                                     vocab,
                                                     VALID_CONV,
                                                     delimiter=" ",
                                                     aug_shift_ms=[])
    sys.stdout.flush()
    # Shuffle labels if required
    if args.shuffle:
        print("Shuffling labels")
        np.random.shuffle(y_train)
        np.random.shuffle(y_valid)
    train_ds = Brain2enDataset(x_train, y_train)
    print("Number of training signals: ", len(train_ds))
    valid_ds = Brain2enDataset(x_valid, y_valid)
    print("Number of validation signals: ", len(valid_ds))
    my_collator = MyCollator(CONFIG, vocab)
    train_dl = data.DataLoader(train_ds,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=CONFIG["num_cpus"],
                               collate_fn=my_collator)
    valid_dl = data.DataLoader(valid_ds,
                               batch_size=args.batch_size,
                               num_workers=CONFIG["num_cpus"],
                               collate_fn=my_collator)

# Default models and parameters
DEFAULT_MODELS = {
    "ConvNet10": (len(vocab), ),
    "PITOM": (len(vocab), len(args.electrodes) * len(args.subjects)),
    "MeNTALmini":
    (len(args.electrodes) * len(args.subjects), len(vocab), args.tf_dmodel,
     args.tf_nhead, args.tf_nlayer, args.tf_dff, args.tf_dropout),
    "MeNTAL":
    (len(args.electrodes) * len(args.subjects), len(vocab), args.tf_dmodel,
     args.tf_nhead, args.tf_nlayer, args.tf_dff, args.tf_dropout)
}

# Create model
if args.init_model is None:
    if args.model in DEFAULT_MODELS:
        print("Building default model: %s" % args.model, end="")
        model_class = globals()[args.model]
        model = model_class(*(DEFAULT_MODELS[args.model]))
    else:
        print("Building custom model: %s" % args.model, end="")
        sys.exit(1)
else:
    model_name = "%s%s.pt" % (SAVE_DIR, args.model)
    if os.path.isfile(model_name):
        model = torch.load(model_name)
        model = model.module if hasattr(model, 'module') else model
        print("Loaded initial model: %s " % args.model)
    else:
        print("No models found in: ", SAVE_DIR)
        sys.exit(1)
print(" with %d trainable parameters" %
      sum([p.numel() for p in model.parameters() if p.requires_grad]))
sys.stdout.flush()

# Initialize loss and optimizer
""" weights = torch.ones(n_classes)
max_freq = -1.
for i in range(n_classes):
    max_freq = max(max_freq, word2freq[vocab[i]])
    weights[i] = 1./float(word2freq[vocab[i]])
weights = weights*max_freq
print(
    sorted([(vocab[i], round(float(weights[i]), 1)) for i in range(n_classes)],
           key=lambda x: x[1])) """
criterion = nn.CrossEntropyLoss()
step_size = int(math.ceil(len(train_ds) / args.batch_size))
optimizer = AdamW(model.parameters(),
                  lr=args.lr,
                  weight_decay=args.weight_decay)
scheduler = None

# Move model and loss to GPUs
if args.gpus:
    model.cuda()
    criterion.cuda()
    if args.gpus > 1:
        model = nn.DataParallel(model)

# Batch chunk size to send to single GPU
# import math
# chunk_size = int(math.ceil(float(args.batch_size)/max(1,args.gpus)))

# Training and evaluation script
if __name__ == "__main__":

    print("Training on %d GPU(s) with batch_size %d for %d epochs" %
          (args.gpus, args.batch_size, args.epochs))
    print("=" * CONFIG["print_pad"])
    sys.stdout.flush()

    best_val_loss = float("inf")
    best_model = model
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    """ train_loss_compute = SimpleLossCompute(criterion,
                                           opt=optimizer,
                                           scheduler=scheduler)
    valid_loss_compute = SimpleLossCompute(criterion, opt=None, scheduler=None)
    """
    epoch = 0
    model_name = "%s%s.pt" % (SAVE_DIR, args.model)
    """ totalfreq = float(sum(train_ds.train_freq.values()))
    print(
        sorted(((i2w[l], f / totalfreq)
                for l, f in train_ds.train_freq.most_common()),
               key=lambda x: -x[1]))
    """
    # Run training and validation for args.epochs epochs
    lr = args.lr
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print('| train | epoch %d | ' % epoch, end='')
        train_loss, train_acc = train(
            train_dl,
            model,
            criterion,
            list(range(args.gpus)),
            optimizer,
            scheduler=scheduler,
            seq2seq=not classify,
            pad_idx=vocab[CONFIG["pad_token"]] if not classify else -1)
        for param_group in optimizer.param_groups:
            if 'lr' in param_group:
                print(' | lr {:1.2E}'.format(param_group['lr']))
                break
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print('| valid | epoch %d | ' % epoch, end='')
        with torch.no_grad():
            valid_loss, valid_acc = valid(
                valid_dl,
                model,
                criterion,
                temperature=args.temp,
                seq2seq=not classify,
                pad_idx=vocab[CONFIG["pad_token"]] if not classify else -1)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        print('|' + '-' * (CONFIG["print_pad"] - 2) + '|')
        # Store best model so far
        if valid_loss < best_val_loss:
            best_model, best_val_loss = model, valid_loss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_to_save = best_model.module\
                    if hasattr(best_model, 'module') else best_model
                torch.save(model_to_save, model_name)
            sys.stdout.flush()

        # if epoch > 10 and valid_loss > max(history['valid_loss'][-3:]):
        #     lr /= 2.
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

    # Plot loss,accuracy vs. time and save figures
    plot_training(history, SAVE_DIR, title="%s_lr%s" % (args.model, args.lr))

    # Save best model found
    # print("Saving best model as %s.pt" % args.model)
    # sys.stdout.flush()

    if not args.no_eval and classify:

        print("Evaluating predictions on test set")
        # Load best model
        model = torch.load(model_name)
        if args.gpus:
            if args.gpus > 1:
                model = nn.DataParallel(model)
            model.to(DEVICE)

        start, end = 0, 0
        softmax = nn.Softmax(dim=1)
        all_preds = np.zeros((x_valid.size(0), n_classes), dtype=np.float32)
        print('Allocating', np.prod(all_preds.shape) * 5 / 1e9, 'GB')

        # Calculate all predictions on test set
        with torch.no_grad():
            for batch in valid_dl:
                src, trg = batch[0].to(DEVICE), batch[1].to(DEVICE,
                                                            dtype=torch.long)
                end = start + src.size(0)
                out = softmax(model(src))
                all_preds[start:end, :] = out.cpu()
                start = end

        print("Calculated predictions")

        # Make categorical
        n_examples = y_valid.shape[0]
        categorical = np.zeros((n_examples, n_classes), dtype=np.float32)
        categorical[np.arange(n_examples), y_valid] = 1

        train_freq = Counter(y_train.tolist())

        # Evaluate top-k
        print("Evaluating top-k")
        sys.stdout.flush()
        res = evaluate_topk(all_preds,
                            y_valid.numpy(),
                            i2w,
                            train_freq,
                            SAVE_DIR,
                            suffix='-val',
                            min_train=args.vocab_min_freq)

        # Evaluate ROC-AUC
        print("Evaluating ROC-AUC")
        sys.stdout.flush()
        res.update(
            evaluate_roc(all_preds,
                         categorical,
                         i2w,
                         train_freq,
                         SAVE_DIR,
                         do_plot=not args.no_plot,
                         min_train=args.vocab_min_freq))
        pprint(res.items())
        print("Saving results")
        with open(SAVE_DIR + "results.json", "w") as fp:
            json.dump(res, fp, indent=4)

if not args.no_eval and not classify:

    print("Evaluating predictions on test set")
    # Load best model
    model = torch.load(model_name)
    if args.gpus:
        model.cuda()

    all_preds, categorical, all_labs = [], [], []
    softmax = nn.Softmax(dim=1)

    # Calculate all predictions on test set
    with torch.no_grad():
        model.eval()
        for batch in valid_dl:
            src, trg_y = batch[0].cuda(), batch[2].long().cuda()
            trg_pos_mask, trg_pad_mask = batch[3].cuda().squeeze(
            ), batch[4].cuda()
            memory = model.encode(src)
            y = torch.zeros(src.size(0), 1, len(vocab)).long().cuda()
            y_sr = torch.zeros(src.size(0), 1, len(vocab)).long().cuda()
            probs = torch.zeros(src.size(0), 1, len(vocab)).long().cuda()
            y[:, :, vocab[CONFIG["begin_token"]]] = 1
            y_sr[:, :, vocab[CONFIG["begin_token"]]] = 1
            for i in range(trg_y.size(1)):
                out = model.decode(memory, y,
                                   trg_pos_mask[:y.size(1), :y.size(1)],
                                   trg_pad_mask[:, :y.size(1)])[:, -1, :]
                out = softmax(out / args.temp)
                temp = torch.zeros(src.size(0), len(vocab)).long().cuda()
                temp = temp.scatter_(1,
                                     torch.argmax(out, dim=1).unsqueeze(-1), 1)
                y = torch.cat([y, temp.unsqueeze(1)], dim=1)
                # probs = torch.cat([probs, out.unsqueeze(1)], dim=1)
                samples = torch.multinomial(out, 20)
                pred = torch.zeros(out.size(0)).long().cuda()
                for j in range(len(samples)):
                    pred[j] = samples[j, torch.argmax(out[j, samples[j]])]
                temp = torch.zeros(pred.size(0), len(vocab)).long().cuda()
                pred = temp.scatter_(1, pred.unsqueeze(-1), 1).unsqueeze(1)
                y_sr = torch.cat([y_sr, pred], dim=1)
            y, y_sr = y[:, 1:, :], y_sr[:, 1:, :]
            idx = (trg_y != vocab[CONFIG["pad_token"]]).nonzero(as_tuple=True)
            lab = trg_y[idx]
            cat = torch.zeros((lab.size(0), len(vocab)),
                              dtype=torch.long).to(lab.device)
            cat = cat.scatter_(1, lab.unsqueeze(-1), 1)
            all_preds.extend(y[idx].cpu().numpy())
            categorical.extend(cat.cpu().numpy())
            all_labs.extend(lab.cpu().numpy())
            print("Output: ",
                  vocab.DecodeIds(torch.argmax(y[0], dim=1).tolist()))
            print("Output_sr: ",
                  vocab.DecodeIds(torch.argmax(y_sr[0], dim=1).tolist()))
            print("Target: ", vocab.DecodeIds(trg_y[0].tolist()))
            print()
            print("Output: ",
                  vocab.DecodeIds(torch.argmax(y[-1], dim=1).tolist()))
            print("Output_sr: ",
                  vocab.DecodeIds(torch.argmax(y_sr[-1], dim=1).tolist()))
            print("Target: ", vocab.DecodeIds(trg_y[-1].tolist()))
            # print("BLEU: ", sentence_bleu([target_sent], predicted_sent))

    all_preds = np.array(all_preds)
    categorical = np.array(categorical)
    all_labs = np.array(all_labs)
    print("Calculated predictions")

    train_freq = train_ds.train_freq
    i2w = {i: vocab.IdToPiece(i) for i in range(len(vocab))}
    markers = [
        CONFIG["begin_token"], CONFIG["end_token"], CONFIG["oov_token"],
        CONFIG["pad_token"]
    ]

    # Evaluate top-k
    print("Evaluating top-k")
    sys.stdout.flush()
    res = evaluate_topk(all_preds,
                        all_labs,
                        i2w,
                        train_freq,
                        SAVE_DIR,
                        suffix='-val',
                        min_train=args.vocab_min_freq,
                        tokens_to_remove=markers)

    # Evaluate ROC-AUC
    print("Evaluating ROC-AUC")
    sys.stdout.flush()
    res.update(
        evaluate_roc(all_preds,
                     categorical,
                     i2w,
                     train_freq,
                     SAVE_DIR,
                     do_plot=not args.no_plot,
                     min_train=args.vocab_min_freq,
                     tokens_to_remove=markers))
    pprint(res.items())
    print("Saving results")
    with open(SAVE_DIR + "results.json", "w") as fp:
        json.dump(res, fp, indent=4)

print("Done!")
