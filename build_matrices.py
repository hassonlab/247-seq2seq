import glob
import sys
from multiprocessing import Pool

import numpy as np

from data_util import (add_begin_end_tokens, calculate_windows_params,
                       generate_wordpairs, get_electrode, remove_duplicates,
                       return_electrode_array, test_for_bad_window,
                       return_examples)


# Build design matrices from conversation directories,
# and process for word classification
def build_design_matrices_classification(
        CONFIG,
        vocab,
        conversations,
        fs=512,
        delimiter=',',
        aug_shift_ms=[-500, -250, 250]):

    datum_suffix = CONFIG["datum_suffix"]
    electrodes = CONFIG["electrodes"]
    window_ms = CONFIG["window_size"]
    shift_ms = CONFIG["shift"]
    bin_ms = CONFIG["bin_size"]
    conv_dirs = CONFIG["CONV_DIRS"]
    subjects = CONFIG["subjects"]

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

    for conversation, suffix, idx in convs[:15]:
        
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
                lambda x: (vocab[x[0]], int(float(x[2])) + start_offset,
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
        conversations,
        fs=512,
        delimiter=',',
        aug_shift_ms=[-500, -250, 250]):

    # extra stuff that happens inside
    begin_token = CONFIG["begin_token"]
    end_token = CONFIG["end_token"]
    exclude_words = CONFIG["exclude_words"]
    datum_suffix = CONFIG["datum_suffix"]
    electrodes = CONFIG["electrodes"]
    window_ms = CONFIG["window_size"]
    shift_ms = CONFIG["shift"]
    bin_ms = CONFIG["bin_size"]
    conv_dirs = CONFIG["CONV_DIRS"]
    subjects = CONFIG["subjects"]

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

    signals, labels, seq_lengths = [], [], []
    for conversation, suffix, idx in convs[0:15]:

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
        if not bigrams:
            print(f'Skipping bad conversation: {conversation}')
            continue
        bigrams = remove_duplicates(bigrams)

        for bigram in bigrams:
            (seq_length, start_onset, end_onset, n_bins) = (
                calculate_windows_params(bigram, signal_param_dict))
            if seq_length <= 0:
                print("bad bi-gram")
                continue
            seq_lengths.append(seq_length)

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
