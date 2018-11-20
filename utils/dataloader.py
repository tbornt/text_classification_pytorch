import re
import pickle

import pandas as pd
from torchtext import data
from torchtext.data import Dataset, Example
from sklearn.model_selection import train_test_split
from utils.utils import str2list


SUPPORTED_EMBEDDING = [
    'charngram.100d',
    'fasttext.en.300d',
    'fasttext.simple.300d',
    'glove.42B.300d',
    'glove.6B.100d',
    'glove.6B.200d',
    'glove.6B.300d',
    'glove.6B.50d',
    'glove.840B.300d',
    'glove.twitter.27B.100d',
    'glove.twitter.27B.200d',
    'glove.twitter.27B.25d',
    'glove.twitter.27B.50d'
]


class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""

    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex


def tokenizer(text):
    text = " ".join(re.findall("[a-zA-Z]+", text))
    return text.split(' ')


def load_data(type, session, **kwargs):
    if type == 'csv':
        return load_csv_data(session, kwargs)


def load_csv_data(session, kwargs):
    is_train = kwargs['is_train']
    if is_train:
        train_file = session['train_file']
        text_column = str2list(session['text_column'])
        if len(text_column) != 1:
            raise Exception('only 1 text column needed, found %d: %s'% (len(text_column), ','.join(text_column)))
        label_column = str2list(session['label_column'])
        use_cols = text_column + label_column
        batch_size = int(session['batch_size'])
        val_ratio = float(session.get('val_ratio', 0))
        fix_length = session.get('fix_length', None)
        validate_file = session.get('validate_file', None)
        pretrained_embedding = session.get('pretrained_embedding', 'glove.6B.100d')
        if fix_length:
            fix_length = int(fix_length)
        sep = session.get('sep', ',')
        train_df = pd.read_csv(train_file, usecols=use_cols, sep=sep)
        if val_ratio > 0 and not validate_file:
            train_df, test_df = train_test_split(train_df, test_size=val_ratio)
        elif validate_file:
            test_df = pd.read_csv(validate_file, usecols=use_cols, sep=sep)
        else:
            test_df = None

        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, fix_length=fix_length)
        LABEL = data.LabelField(sequential=False, use_vocab=False)
        fields = {}
        for column in text_column:
            fields[column] = TEXT
        for column in label_column:
            fields[column] = LABEL
        train_dataset = DataFrameDataset(train_df, fields=fields)
        if test_df is not None:
            test_dataset = DataFrameDataset(test_df, fields=fields)
        else:
            test_dataset = None
        TEXT.build_vocab(train_dataset, test_dataset)
        if pretrained_embedding not in SUPPORTED_EMBEDDING:
            raise Exception("%s is not supported embedding" % pretrained_embedding)
        TEXT.vocab.load_vectors(pretrained_embedding)
        train_iter = data.BucketIterator(train_dataset,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         repeat=False)
        test_iter = data.BucketIterator(test_dataset,
                                        shuffle=False,
                                        batch_size=batch_size,
                                        repeat=False)
        return train_iter, test_iter, TEXT
    else:
        decode_file = session['decode_file']
        vocab_file = session['vocab_file']
        text_column = str2list(session['text_column'])
        if len(text_column) != 1:
            raise Exception('only 1 text column needed, found %d: %s'% (len(text_column), ','.join(text_column)))
        batch_size = int(session['batch_size'])
        fix_length = session.get('fix_length', None)
        if fix_length:
            fix_length = int(fix_length)

        vocab = pickle.load(open(vocab_file, 'rb'))
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, fix_length=fix_length)
        TEXT.vocab = vocab
        fields = {}
        for column in text_column:
            fields[column] = TEXT

        decode_df = pd.read_csv(decode_file, usecols=text_column)
        decode_dataset = DataFrameDataset(decode_df, fields=fields)
        decode_iter = data.BucketIterator(decode_dataset,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         repeat=False)
        return decode_iter, TEXT
