import glob
import os
import sys
from collections import Counter

import pandas as pd
import sentencepiece as spm


# Build vocabulary by reading datums
def get_vocab(conv_dirs,
              subjects,
              min_freq=1,
              exclude_words=['sp', '{lg}', '{ns}'],
              datum_suffix=["conversation_trimmed", "trimmed"]):
    # global word2freq, vocab, n_classes, w2i, i2w
    exclude_words = set(exclude_words)
    word2freq = Counter()
    columns = ["word", "onset", "offset", "accuracy", "speaker"]
    files = [
        f for conv_dir, subject, ds in zip(conv_dirs, subjects, datum_suffix)
        for f in glob.glob(conv_dir + f'NY{subject}*/misc/*datum_{ds}.txt')
    ]
    for filename in files:
        df = pd.read_csv(filename, delimiter=' ', header=None, names=columns)
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
    print("# Conversations:", len(files))
    print("Vocabulary size (min_freq=%d): %d" % (min_freq, len(word2freq)))
    return word2freq, vocab, n_classes, w2i, i2w


# Build vocabulary by using sentencepiece (seq2seq required)
def get_sp_vocab(CONFIG,
                 conv_dirs,
                 subjects,
                 algo='unigram',
                 vocab_size=1000,
                 exclude_words=['sp', '{lg}', '{ns}'],
                 datum_suffix=["conversation_trimmed", "trimmed"],
                 oov_tok="<unk>",
                 begin_tok="<s>",
                 end_tok="</s>",
                 pad_tok="<pad>"):
    exclude_words = set(exclude_words)
    columns = ["word", "onset", "offset", "accuracy", "speaker"]
    files = [
        f for conv_dir, subject, ds in zip(conv_dirs, subjects, datum_suffix)
        for f in glob.glob(conv_dir + f'NY{subject}*/misc/*datum_{ds}.txt')
    ]
    words = []
    for filename in files:
        df = pd.read_csv(filename, delimiter=' ', header=None, names=columns)
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
    print("# Conversations:", len(files))
    print("Vocabulary size (%s): %d" % (algo, vocab_size))
    return vocab
