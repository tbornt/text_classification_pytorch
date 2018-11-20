import os
import errno
import argparse

import pandas as pd


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-t', '--test', required=True, help='test file')
arg_parser.add_argument('--decode_result', required=True, help='identity hate result')
arg_parser.add_argument('-o', '--output', required=True, help='output file')


def load_df(file_path, names):
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    df = pd.read_csv(file_path, header=None, names=names)
    return df


if __name__ == '__main__':
    args = arg_parser.parse_args()
    test_file = args.test
    if not os.path.exists(test_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), test_file)
    test_df = pd.read_csv(test_file, usecols=['id'])

    decode_file = args.decode_result
    decode_df = pd.read_csv(decode_file, ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

    res = pd.concat([test_df['id'], decode_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]], axis=1)
    res.to_csv(args.output, index=False)
