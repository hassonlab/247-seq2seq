import argparse


def arg_parser():
    ''' Explanation of the Arguments:
    model (string): Type of model to run from among
                    (PITOM, ConvNet, MeNTALmini, MeNTAL)
    subject (list of strings): subject id's as a list
    shift (integer): Amount by which the onset should be shifted
    lr (float): learning rate
    gpus (int): number of gpus for the model to run on
    epochs (int):
    batch-size (int):
    window-size (int): window size to consider for the word in ms
    bin-size (int): bin size in ms
    init-model (string):
'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MeNTAL')
    parser.add_argument('--subjects', nargs='*', default=['625', '676'])
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--window-size', type=int, default=2000)
    parser.add_argument('--bin-size', type=int, default=50)
    parser.add_argument('--init-model', type=str, default=None)
    parser.add_argument('--no-plot', action='store_false', default=False)
    parser.add_argument('--electrodes', nargs='*', default=list(range(1, 65)))
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--shuffle', action="store_true", default=False)
    parser.add_argument('--no-eval', action="store_true", default=False)
    parser.add_argument('--temp', type=float, default=0.995)
    parser.add_argument('--tf-dmodel', type=int, default=64)
    parser.add_argument('--tf-dff', type=int, default=256)
    parser.add_argument('--tf-nhead', type=int, default=8)
    parser.add_argument('--tf-nlayer', type=int, default=12)
    parser.add_argument('--tf-dropout', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.35)
    args = parser.parse_args()

    return args
