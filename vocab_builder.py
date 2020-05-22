import glob
import json
import os
import sys
from collections import Counter, OrderedDict

import pandas as pd
import sentencepiece as spm

from data_util import return_conversations


def get_std_vocab(CONFIG, comprehension=True, classify=True):
    # Build vocabulary by reading datums
    min_freq = CONFIG["vocab_min_freq"]
    max_freq = CONFIG["vocab_max_freq"]
    exclude_words = set(CONFIG["exclude_words"])
    word2freq = Counter()

    convs = return_conversations(CONFIG, 'train')

    tokens_to_add = [
        CONFIG["begin_token"], CONFIG["end_token"], CONFIG["oov_token"],
        CONFIG["pad_token"]
    ]

    start_index = 0 if classify else len(tokens_to_add)

    conv_count = 0
    for conversation, suffix, _, _ in convs:
        datum_fn = glob.glob(conversation + suffix)[0]
        if not datum_fn:
            print('File DNE: ', conversation + suffix)
            continue
        conv_count += 1

        with open(datum_fn, 'r') as fin:
            lines = map(lambda x: x.split(), fin)
            examples = list(map(
                lambda x: (" ".join([
                    z for y in x[0:-4] if (z:= y.lower().strip().replace(
                        '"', '')) not in exclude_words
                ]), x[-1]), lines))
            if not comprehension:
                examples = [x[0] for x in examples if x[1] == 'Speaker1']
            else:
                examples = [x[0] for x in examples]
            examples = filter(lambda x: len(x) > 0, examples)
            examples = list(map(lambda x: x.split(), examples))
        word2freq.update(word for example in examples for word in example)

    if min_freq > 1:
        word2freq = dict(filter(lambda x: x[1] >= min_freq and x[1] <= max_freq, word2freq.items()))

    vocab = sorted(word2freq.keys())
    w2i = {word: i for i, word in enumerate(vocab, start_index)}

    if not classify:
        w2f_tok, w2i_tok = {}, {}
        for i, token in enumerate(tokens_to_add):
            w2f_tok[token] = -1
            w2i_tok[token] = i
        word2freq.update(w2f_tok)
        w2i.update(w2i_tok)
        vocab.extend(tokens_to_add)
        w2i = OrderedDict(sorted(w2i.items(), key=lambda kv: kv[1]))

    n_classes = len(vocab)
    i2w = {i: word for word, i in w2i.items()}

    print("# Conversations:", conv_count)
    print("Vocabulary size (min_freq=%d): %d" % (min_freq, len(word2freq)))

    # Save word counter
    print("Saving word counter")
    write_df = pd.DataFrame.from_dict(word2freq, orient='index', columns=['Frequency'])
    write_df['Word'] = write_df.index
    write_df = write_df[['Word', 'Frequency']]
    write_df.to_excel(os.path.join(CONFIG["SAVE_DIR"], "word2freq.xlsx"), index=False)

    # figure1(save_dir, word2freq)
    # figure2(save_dir, word2freq)
    # figure3(save_dir, word2freq)

    return word2freq, vocab, n_classes, w2i, i2w


# Build vocabulary by reading datums
def get_vocab(CONFIG):

    min_freq = CONFIG["vocab_min_freq"]
    exclude_words = set(CONFIG["exclude_words_class"])
    word2freq = Counter()
    columns = ["word", "onset", "offset", "accuracy", "speaker"]

    convs = return_conversations(CONFIG, 'train')

    conv_count = 0
    for conversation, suffix, _, _ in convs:
        datum_fn = glob.glob(conversation + suffix)[0]
        if not datum_fn:
            print('File DNE: ', conversation + suffix)
            continue

        conv_count += 1
        df = pd.read_csv(datum_fn, delimiter=' ', header=None, names=columns)
        df.word = df.word.str.lower()
        df = df[df.speaker == "Speaker1"]
        # Exclude specified words and double words
        word2freq.update(word for word in df.word.tolist()
                         if (word not in exclude_words) and (" " not in word))
    if min_freq > 1:
        word2freq = {
            word: freq
            for word, freq in word2freq.items() if freq >= min_freq
        }
    vocab = sorted(word2freq.keys())
    n_classes = len(vocab)
    w2i = {word: i for i, word in enumerate(vocab)}
    i2w = {i: word for word, i in w2i.items()}

    print("# Conversations:", conv_count)
    print("Vocabulary size (min_freq=%d): %d" % (min_freq, len(word2freq)))

    # Save word counter
    print("Saving word counter")
    write_df = pd.DataFrame.from_dict(word2freq, orient='index', columns=['Frequency'])
    write_df['Word'] = write_df.index
    write_df = write_df[['Word', 'Frequency']]
    write_df.to_excel(os.path.join(CONFIG["SAVE_DIR"], "word2freq.xlsx"), index=False)
    
    return word2freq, vocab, n_classes, w2i, i2w


# Build vocabulary by using sentencepiece (seq2seq required)
def get_sp_vocab(CONFIG, algo='unigram', vocab_size=1000):
    exclude_words = set(CONFIG["exclude_words"])
    columns = ["word", "onset", "offset", "accuracy", "speaker"]
    oov_tok = CONFIG["oov_token"]

    convs = return_conversations(CONFIG, 'train')

    words, conv_count = [], 0
    for conversation, suffix, idx, _ in convs:

        # Check if files exists, if it doesn't go to next
        datum_fn = glob.glob(conversation + suffix)[0]
        if not datum_fn:
            print('File DNE: ', conversation + suffix)
            continue

        conv_count += 1
        df = pd.read_csv(datum_fn, delimiter=' ', header=None, names=columns)
        df.word = df.word.str.lower()
        df = df[df.speaker == "Speaker1"]
        for e in exclude_words:
            df.word = df[df.word.str.lower() != e]
        words.append(df.word.dropna().tolist())

    vocab_file = os.path.join(CONFIG["SAVE_DIR"], "vocab_temp.txt")
    with open(vocab_file, "w") as f:
        for conv in words:
            for line in conv:
                f.write(str(line) + '\n')
    spm.SentencePieceTrainer.Train(
        '--input=%s --model_prefix=MeNTAL --model_type=%s \
            --vocab_size=%d --bos_id=0 --eos_id=1 --unk_id=2 --unk_surface=%s\
                --pad_id=3' % (vocab_file, algo, vocab_size, oov_tok))
    sys.stdout.flush()
    vocab = spm.SentencePieceProcessor()
    vocab.Load("MeNTAL.model")

    print("# Conversations:", conv_count)
    print("Vocabulary size (%s): %d" % (algo, vocab_size))

    return vocab
