import os
import re
import pandas as pd


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"\. \. \.", "\.", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__ == '__main__':
    data_dir = '../../data/sst'
    file_names = ['stsa.fine.dev', 'stsa.fine.test', 'stsa.fine.train']

    labels = []
    sentences = []
    for file_name in file_names:
        with open(os.path.join(data_dir, file_name)) as f:
            for line in f:
                div = line.index(' ')
                sentences.append(clean_str(line[div + 1:]))
                labels.append(line[:div])
        df = pd.DataFrame({'text': sentences, 'label': labels})
        df.to_csv(os.path.join(data_dir, 'df_'+file_name), index=False, sep='\t')
