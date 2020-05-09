import glob
import math
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from scipy.io import loadmat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Read file line-by-line and store in list
def read_file(fn):
    with open(fn, 'r') as f:
        lines = [line.rstrip() for line in f]
    return lines


# # Build vocabulary by reading datums
# def get_vocab(conv_dirs,
#               subjects,
#               min_freq=1,
#               exclude_words=['sp', '{lg}', '{ns}'],
#               datum_suffix=["conversation_trimmed", "trimmed"]):
#     # global word2freq, vocab, n_classes, w2i, i2w
#     exclude_words = set(exclude_words)
#     word2freq = Counter()
#     columns = ["word", "onset", "offset", "accuracy", "speaker"]
#     files = [
#         f for conv_dir, subject, ds in zip(conv_dirs, subjects, datum_suffix)
#         for f in glob.glob(conv_dir + f'NY{subject}*/misc/*datum_{ds}.txt')
#     ]
#     for filename in files:
#         df = pd.read_csv(filename, delimiter=' ', header=None, names=columns)
#         df.word = df.word.str.lower()
#         df = df[df.speaker == "Speaker1"]
#         # Exclude specified words and double words
#         word2freq.update(word for word in df.word.tolist()
#                          if (word not in exclude_words) and (" " not in word))
#     if min_freq > 1:
#         word2freq = {
#             word: freq
#             for word, freq in word2freq.items() if freq >= min_freq
#         }
#     vocab = sorted(word2freq.keys())
#     n_classes = len(vocab)
#     w2i = {word: i for i, word in enumerate(vocab)}
#     i2w = {i: word for word, i in w2i.items()}
#     print("# Conversations:", len(files))
#     print("Vocabulary size (min_freq=%d): %d" % (min_freq, len(word2freq)))
#     return word2freq, vocab, n_classes, w2i, i2w


# # Build vocabulary by using sentencepiece (seq2seq required)
# def get_sp_vocab(CONFIG,
#                  conv_dirs,
#                  subjects,
#                  algo='unigram',
#                  vocab_size=1000,
#                  exclude_words=['sp', '{lg}', '{ns}'],
#                  datum_suffix=["conversation_trimmed", "trimmed"],
#                  oov_tok="<unk>",
#                  begin_tok="<s>",
#                  end_tok="</s>",
#                  pad_tok="<pad>"):
#     exclude_words = set(exclude_words)
#     columns = ["word", "onset", "offset", "accuracy", "speaker"]
#     files = [
#         f for conv_dir, subject, ds in zip(conv_dirs, subjects, datum_suffix)
#         for f in glob.glob(conv_dir + f'NY{subject}*/misc/*datum_{ds}.txt')
#     ]
#     words = []
#     for filename in files:
#         df = pd.read_csv(filename, delimiter=' ', header=None, names=columns)
#         df.word = df.word.str.lower()
#         df = df[df.speaker == "Speaker1"]
#         for e in exclude_words:
#             df.word = df[df.word.str.lower() != e]
#         words.append(df.word.dropna().tolist())

#     vocab_file = os.path.join(CONFIG["SAVE_DIR"], "vocab_temp.txt")
#     with open(vocab_file, "w") as f:
#         for conv in words:
#             for line in conv:
#                 f.write(str(line) + '\n')
#     spm.SentencePieceTrainer.Train(
#         '--input=%s --model_prefix=MeNTAL --model_type=%s \
#             --vocab_size=%d --bos_id=0 --eos_id=1 --unk_id=2 --unk_surface=%s\
#                 --pad_id=3' % (vocab_file, algo, vocab_size, oov_tok))
#     sys.stdout.flush()
#     vocab = spm.SentencePieceProcessor()
#     vocab.Load("MeNTAL.model")
#     print("# Conversations:", len(files))
#     print("Vocabulary size (%s): %d" % (algo, vocab_size))
#     return vocab


# Get electrode date helper
def get_electrode(elec_id):
    conversation, electrode = elec_id
    search_str = conversation + f'/preprocessed/*_{electrode}.mat'
    mat_fn = glob.glob(search_str)
    if len(mat_fn) == 0:
        print(f'[WARNING] electrode {electrode} DNE in {search_str}')
        return None
    return loadmat(mat_fn[0])['p1st'].squeeze().astype(np.float32)


def return_electrode_array(conv, elect):
    # Read signals
    elec_ids = ((conv, electrode) for electrode in elect)
    with Pool() as pool:
        ecogs = list(
            filter(lambda x: x is not None, pool.map(get_electrode, elec_ids)))

    ecogs = np.asarray(ecogs)
    ecogs = (ecogs - ecogs.mean(axis=1).reshape(
        ecogs.shape[0], 1)) / ecogs.std(axis=1).reshape(ecogs.shape[0], 1)
    ecogs = ecogs.T
    assert (ecogs.ndim == 2 and ecogs.shape[1] == len(elect))
    return ecogs


def return_examples(file, delim, vocabulary, ex_words):
    with open(file, 'r') as fin:
        lines = map(lambda x: x.split(delim), fin)
        examples = map(
            lambda x: (" ".join([
                z for y in x[0:-4]
                if (z:= y.lower().strip().replace('"', '')) not in ex_words
            ]), x[-1].strip() == "Speaker1", x[-4], x[-3]), lines)
        examples = filter(lambda x: len(x[0]) > 0, examples)
        examples = map(
            lambda x: (vocabulary.EncodeAsIds(x[0]), x[1], int(float(x[2])),
                       int(float(x[3]))), examples)
        return list(examples)


def generate_wordpairs(examples):
    # if the first set already has two words and is speaker 1
    # if the second set already has two words and is speaker 1
    # the onset of the first word is earlier than the second word
    my_grams = []
    for first, second in zip(examples, examples[1:]):
        len1, len2 = len(first[0]), len(second[0])
        if first[1] and len1 == 2:
            my_grams.append(first)
        if second[1] and len2 == 2:
            my_grams.append(second)
        if first[1] and second[1]:
            if len1 == 1 and len2 == 1:
                if first[2] < second[2]:
                    ak = (first[0] + second[0], True, first[2], second[3])
                    my_grams.append(ak)
    return my_grams


def remove_duplicates(grams):
    df = pd.DataFrame(grams)
    df[['fw', 'sw']] = pd.DataFrame(df[0].tolist())
    df = df.drop(columns=[0]).drop_duplicates()
    df[0] = df[['fw', 'sw']].values.tolist()
    df = df.drop(columns=['fw', 'sw'])
    df = df[sorted(df.columns)]
    return list(df.to_records(index=False))


def add_begin_end_tokens(word_pair, vocabulary, start_tok, stop_tok):
    word_pair.insert(0, vocabulary[start_tok])  # Add start token
    word_pair.append(vocabulary[stop_tok])  # Add end token
    return word_pair


def test_for_bad_window(start, stop, shape, window):
    # if the window_begin is less than 0 or
    # check if onset is within limits
    # if the window_end is less than 0 or
    # if the window_end is outside the signal
    # if there are not enough frames in the window
    return (start < 0 or start > shape[0] or stop < 0 or stop > shape[0]
            or stop - start < window)


def calculate_windows_params(gram, param_dict):
    begin_window = gram[2] + param_dict['start_offset']
    end_window = gram[3] + param_dict['end_offset']
    bin_size = int(
        math.ceil((end_window - begin_window) /
                  param_dict['bin_fs']))  # calculate number of bins

    return begin_window, end_window, bin_size


# Build design matrices from conversation directories,
# and process for word classification
def build_design_matrices_classification(
        conv_dirs,
        subjects,
        conversations,
        fs=512,
        bin_ms=50,
        shift_ms=0,
        window_ms=2000,
        delimiter=',',
        electrodes=[],
        datum_suffix=["conversation_trimmed", "trimmed"],
        aug_shift_ms=[-500, -250, 250]):
    global w2i
    signals, labels = [], []
    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    start_offset = -half_window + shift_fs
    end_offset = half_window + shift_fs
    n_bins = len(range(-half_window, half_window, bin_fs))
    convs = [
        (conv_dir + conv_name, '/misc/*datum_%s.txt' % ds, idx)
        for idx, (conv_dir, convs,
                  ds) in enumerate(zip(conv_dirs, conversations, datum_suffix))
        for conv_name in convs
    ]
    aug_shift_fs = [int(s / 1000 * fs) for s in aug_shift_ms]

    for conversation, suffix, idx in convs:
        datum_fn = glob.glob(conversation + suffix)
        if len(datum_fn) == 0:
            print('File DNE: ', conversation + suffix)
            continue
        datum_fn = datum_fn[0]

        # Read signals
        ecogs = []
        elec_ids = ((conversation, electrode) for electrode in electrodes)
        with Pool() as pool:
            ecogs = list(
                filter(lambda x: x is not None,
                       pool.map(get_electrode, elec_ids)))
        if len(ecogs) == 0:
            print(f'Skipping bad conversation: {conversation}')
            continue
        ecogs = np.asarray(ecogs).T
        assert (ecogs.ndim == 2 and ecogs.shape[1] == len(electrodes))

        # Read conversations and form examples
        max_example_idx = ecogs.shape[0]
        old_size = len(signals)
        examples = []
        with open(datum_fn, 'r') as fin:
            lines = map(lambda x: x.split(delimiter), fin)
            examples = map(
                lambda x:
                (x[0].lower().strip().replace('"', ''), x[4].strip(), x[1]),
                lines)
            examples = filter(lambda x: x[0] in vocab and x[1] == "Speaker1",
                              examples)
            examples = map(
                lambda x: (w2i[x[0]], int(float(x[2])) + start_offset,
                           int(float(x[2])) + end_offset), examples)
            examples = filter(
                lambda x: x[1] >= 0 and x[1] <= max_example_idx and x[2] >= 0
                and x[2] <= max_example_idx, examples)
            word_signal = np.zeros((n_bins, len(electrodes) * len(subjects)),
                                   np.float32)
            for x in examples:
                labels.append(x[0])
                for i, f in enumerate(
                        np.array_split(ecogs[x[1]:x[2], :], n_bins, axis=0)):
                    word_signal[i, idx * len(electrodes):(idx + 1) *
                                len(electrodes)] = f.mean(axis=0)
                signals.append(np.copy(word_signal))
                # Data augmentation by shifts
                for i, s in enumerate(aug_shift_fs):
                    aug_start = x[1] + s
                    if aug_start < 0 or aug_start > max_example_idx:
                        continue
                    aug_end = x[2] + s
                    if aug_end < 0 or aug_end > max_example_idx:
                        continue
                    for i, f in enumerate(
                            np.array_split(ecogs[aug_start:aug_end, :],
                                           n_bins,
                                           axis=0)):
                        word_signal[i, idx * len(electrodes):(idx + 1) *
                                    len(electrodes)] = f.mean(axis=0)
                    signals.append(np.copy(word_signal))
                    labels.append(x[0])

        if len(signals) == old_size:
            print(f'[WARNING] no examples built for {conversation}')

    if len(signals) == 0:
        print('[ERROR] signals is empty')
        sys.exit(1)

    return np.array(signals), np.array(labels)


def build_design_matrices_seq2seq(
        CONFIG,
        vocab,
        conv_dirs,
        subjects,
        conversations,
        fs=512,
        bin_ms=50,
        shift_ms=0,
        window_ms=2000,
        delimiter=',',
        electrodes=[],
        datum_suffix=["conversation_trimmed", "trimmed"],
        exclude_words=['sp', '{lg}', '{ns}'],
        aug_shift_ms=[-500, -250, 250]):

    # extra stuff that happens inside
    begin_token = CONFIG["begin_token"]
    end_token = CONFIG["end_token"]

    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    start_offset = -half_window + shift_fs
    end_offset = half_window + shift_fs

    signal_param_dict = dict()
    signal_param_dict['bin_fs'] = bin_fs
    signal_param_dict['shift_fs'] = shift_fs
    signal_param_dict['window_fs'] = window_fs
    signal_param_dict['half_window'] = half_window
    signal_param_dict['start_offset'] = start_offset
    signal_param_dict['end_offset'] = end_offset

    convs = [
        (conv_dir + conv_name, '/misc/*datum_%s.txt' % ds, idx)
        for idx, (conv_dir, convs,
                  ds) in enumerate(zip(conv_dirs, conversations, datum_suffix))
        for conv_name in convs
    ]

    signals, labels = [], []
    for conversation, suffix, idx in convs[0:10]:

        # Check if files exists, if it doesn't go to next
        datum_fn = glob.glob(conversation + suffix)[0]
        if not datum_fn:
            print('File DNE: ', conversation + suffix)
            continue

        # Extract electrode data
        ecogs = return_electrode_array(conversation, electrodes)
        if not ecogs.size:
            print(f'Skipping bad conversation: {conversation}')
            continue

        examples = return_examples(datum_fn, delimiter, vocab, exclude_words)
        bigrams = generate_wordpairs(examples)
        bigrams = remove_duplicates(bigrams)

        for bigram in bigrams:
            start_onset, end_onset, n_bins = calculate_windows_params(
                bigram, signal_param_dict)

            if test_for_bad_window(start_onset, end_onset, ecogs.shape,
                                   window_fs):
                continue

            labels.append(
                add_begin_end_tokens(bigram[0], vocab, begin_token, end_token))
            word_signal = np.zeros((n_bins, len(electrodes) * len(subjects)),
                                   np.float32)
            for i, f in enumerate(
                    np.array_split(ecogs[start_onset:end_onset, :],
                                   n_bins,
                                   axis=0)):
                word_signal[i, idx * len(electrodes):(idx + 1) *
                            len(electrodes)] = f.mean(axis=0)

            # TODO: Data Augmentation
            signals.append(word_signal)
    print('final')
    assert len(labels) == len(signals), "Bad Shape for Lengths"
    return signals, labels


# Pytorch Dataset wrapper
class Brain2enDataset(Dataset):
    """Brainwave-to-English Dataset."""
    def __init__(self, signals, labels):
        """
        Args:
            signals (list): brainwave examples.
            labels (list): english examples.
        """
        global oov_token, vocab

        assert (len(signals) == len(labels))
        indices = [(i, len(signals[i]), len(labels[i]))
                   for i in range(len(signals))]
        indices.sort(key=lambda x: (x[1], x[2], x[0]))
        self.examples = []
        self.max_seq_len = 0
        self.max_sent_len = 0
        self.train_freq = Counter()
        c = 0
        for i in indices:
            if i[1] > 384 or i[2] < 4 or i[2] > 128:
                c += 1
                continue
            lab = labels[i[0]]
            self.train_freq.update(lab)
            lab = torch.tensor(lab).long()
            self.examples.append(
                (torch.from_numpy(signals[i[0]]).float(), lab))
            self.max_seq_len = max(self.max_seq_len, i[1])
            self.max_sent_len = max(self.max_sent_len, len(lab))
        print("Skipped", c, "examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MyCollator(object):
    def __init__(self, CONFIG, vocabulary):
        self.CONFIG = CONFIG
        self.vocabulary = vocabulary
        self.pad_token = CONFIG["pad_token"]

    def __call__(self, batch):
        # do something with batch and self.params
        src = pad_sequence([batch[i][0] for i in range(len(batch))],
                           batch_first=True,
                           padding_value=0.)
        labels = pad_sequence([batch[i][1] for i in range(len(batch))],
                              batch_first=True,
                              padding_value=self.vocabulary[self.pad_token])
        trg = torch.zeros(labels.size(0), labels.size(1),
                          len(self.vocabulary)).scatter_(
                              2, labels.unsqueeze(-1), 1)
        trg, trg_y = trg[:, :-1, :], labels[:, 1:]
        pos_mask, pad_mask = self.masks(trg_y)
        return src, trg, trg_y, pos_mask, pad_mask

    def masks(self, labels):
        pos_mask = (torch.triu(torch.ones(labels.size(1),
                                          labels.size(1))) == 1).transpose(
                                              0, 1).unsqueeze(0)
        pos_mask = pos_mask.float().masked_fill(pos_mask == 0,
                                                float('-inf')).masked_fill(
                                                    pos_mask == 1, float(0.0))
        pad_mask = labels == self.vocabulary[self.pad_token]
        return pos_mask, pad_mask
