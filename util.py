import os
from argparse import ArgumentParser

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_train_args():
    parser = ArgumentParser(description='Bitext mining using dual encoder LSTM')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=320)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--no-bidirectional', action='store_false', dest='bidirectional')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args


def get_scorer_args():
    parser = ArgumentParser(description='Bitext mining using dual encoder LSTM')
    parser.add_argument('--input', type=str, default='test.tsv')
    parser.add_argument('--output', type=str, default='scores.txt')
    parser.add_argument('--model', type=str, default='model.pt')
    parser.add_argument('--vocab', type=str, default='vocab.pt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args
