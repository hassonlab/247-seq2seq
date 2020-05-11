import glob
import json
import os
import sys
from collections import Counter

import pandas as pd
import sentencepiece as spm

from data_util import return_conversations


# Build vocabulary by reading datums
def get_vocab(CONFIG):

    min_freq = CONFIG["vocab_min_freq"]
    exclude_words = set(CONFIG["exclude_words_class"])
    word2freq = Counter()
    columns = ["word", "onset", "offset", "accuracy", "speaker"]
    conversations = CONFIG["TRAIN_CONV"]
    
    convs = return_conversations(CONFIG, conversations)

    conv_count = 0
    for conversation, suffix, _ in convs:
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
    with open("%sword2freq.json" % CONFIG["SAVE_DIR"], "w") as fp:
        json.dump(word2freq, fp, indent=4)
    sys.stdout.flush()

    return word2freq, vocab, n_classes, w2i, i2w


# Build vocabulary by using sentencepiece (seq2seq required)
def get_sp_vocab(CONFIG, algo='unigram', vocab_size=1000):
    exclude_words = set(CONFIG["exclude_words"])
    columns = ["word", "onset", "offset", "accuracy", "speaker"]
    conversations = CONFIG["TRAIN_CONV"]
    oov_tok = CONFIG["oov_token"]

    convs = return_conversations(CONFIG, conversations)

    words, conv_count = [], 0
    for conversation, suffix, idx in convs[0:10]:

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
