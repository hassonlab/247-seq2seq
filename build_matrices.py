import glob
import sys

import numpy as np

from data_util import (add_begin_end_tokens, calculate_windows_params,
                       convert_ms_to_fs, generate_wordpairs, remove_duplicates,
                       remove_oovs, return_conversations, return_examples,
                       test_for_bad_window)
from electrode_utils import return_electrode_array


# Build design matrices from conversation directories,
# and process for word classification
def build_design_matrices_classification(set_str, CONFIG,
                                         vocab,
                                         fs=512,
                                         delimiter=',',
                                         aug_shift_ms=[-500, -250, 250]):

    electrodes = CONFIG["electrodes"]
    subjects = CONFIG["subjects"]
    signal_param_dict = convert_ms_to_fs(CONFIG)

    half_window = signal_param_dict["half_window"]
    n_bins = len(range(-half_window, half_window, signal_param_dict["bin_fs"]))

    convs = return_conversations(CONFIG, set_str)
    aug_shift_fs = [int(s / 1000 * fs) for s in aug_shift_ms]

    signals, labels = [], []
    for conversation, suffix, idx in convs:

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
                lambda x: (vocab[x[0]], int(
                    float(x[2])) + signal_param_dict["start_offset"],
                           int(float(x[2])) + signal_param_dict["end_offset"]),
                examples)
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


def build_design_matrices_seq2seq(set_str,
                                  CONFIG,
                                  vocab,
                                  fs=512,
                                  delimiter=',',
                                  aug_shift_ms=[-500, -250, 250]):

    # extra stuff that happens inside
    begin_token = CONFIG["begin_token"]
    end_token = CONFIG["end_token"]
    exclude_words = CONFIG["exclude_words"]
    electrodes = CONFIG["electrodes"]
    subjects = CONFIG["subjects"]
    signal_param_dict = convert_ms_to_fs(CONFIG)

    convs = return_conversations(CONFIG, set_str)

    signals, labels, seq_lengths = [], [], []
    for conversation, suffix, idx in convs:

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

        examples = return_examples(datum_fn, delimiter, vocab, exclude_words,
                                   CONFIG["vocabulary"])
        bigrams = generate_wordpairs(examples)

        if not bigrams:
            print(f'Skipping bad conversation: {conversation}')
            continue
        bigrams = remove_duplicates(bigrams)
        bigrams = remove_oovs(bigrams, vocab, data_tag=set_str)

        for bigram in bigrams:
            (seq_length, start_onset, end_onset,
             n_bins) = (calculate_windows_params(bigram, signal_param_dict))

            if seq_length <= 0:
                print("bad bi-gram")
                continue
            seq_lengths.append(seq_length)

            if test_for_bad_window(start_onset, end_onset, ecogs.shape,
                                   signal_param_dict['window_fs']):
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

    print(f'Number of {set_str} samples is: {len(signals)}')
    print(f'Number of {set_str} labels is: {len(labels)}')

    print(f'Maximum Sequence Length is: {max([len(i) for i in signals])}')

    assert len(labels) == len(signals), "Bad Shape for Lengths"

    return signals, labels
