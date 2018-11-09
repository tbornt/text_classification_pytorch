import re

import pandas as pd
from torchtext import data
from torchtext.data import Dataset, Example
from sklearn.model_selection import train_test_split


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
        text_column = session['text_column']
        label_column = session['label_column']
        batch_size = int(session['batch_size'])
        val_ratio = float(session.get('val_ratio', 0))

        train_df = pd.read_csv(train_file, usecols=[text_column, label_column])
        if val_ratio > 0:
            train_df, test_df = train_test_split(train_df, test_size=val_ratio)
        else:
            test_df = None

        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)
        LABEL = data.LabelField(sequential=False, use_vocab=False)
        train_dataset = DataFrameDataset(train_df, fields={text_column: TEXT, label_column: LABEL})
        if test_df is not None:
            test_dataset = DataFrameDataset(test_df, fields={text_column: TEXT, label_column: LABEL})
        else:
            test_dataset = None
        TEXT.build_vocab(train_dataset, test_dataset)
        TEXT.vocab.load_vectors('glove.6B.100d')
        train_iter = data.BucketIterator(train_dataset,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         repeat=False)
        test_iter = data.BucketIterator(test_dataset,
                                        shuffle=False,
                                        batch_size=batch_size,
                                        repeat=False)
        return train_iter, test_iter, TEXT
