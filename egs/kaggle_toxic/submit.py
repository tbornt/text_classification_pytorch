import os
import errno
import argparse

import pandas as pd


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-t', '--test', required=True, help='test file')
arg_parser.add_argument('--toxic', required=True, help='toxic result')
arg_parser.add_argument('--severetoxic', required=True, help='severe toxic result')
arg_parser.add_argument('--obscene', required=True, help='obscene result')
arg_parser.add_argument('--threat', required=True, help='threat result')
arg_parser.add_argument('--insult', required=True, help='insult result')
arg_parser.add_argument('--identityhate', required=True, help='identity hate result')
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

    toxic_df = load_df(args.toxic, ['non-toxic', 'toxic'])
    severe_toxic_df = load_df(args.severetoxic, ['non-severe_toxic', 'severe_toxic'])
    obscene_df = load_df(args.obscene, ['non-obscene', 'obscene'])
    threat_df = load_df(args.threat, ['non-threat', 'threat'])
    insult_df = load_df(args.insult, ['non-insult', 'insult'])
    identity_hate_df = load_df(args.identityhate, ['non-identity_hate', 'identity_hate'])

    res = pd.concat([test_df['id'],
                     toxic_df['toxic'],
                     severe_toxic_df['severe_toxic'],
                     obscene_df['obscene'],
                     threat_df['threat'],
                     insult_df['insult'],
                     identity_hate_df['identity_hate']], axis=1)
    res.to_csv(args.output, index=False)
