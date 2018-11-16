import os
import pandas as pd


def score2label(score):
    if 0 <= score <= 0.2:
        return 0
    elif 0.2 < score <= 0.4:
        return 1
    elif 0.4 < score <= 0.6:
        return 2
    elif 0.6 < score <= 0.8:
        return 3
    else:
        return 4

if __name__ == '__main__':
    data_dir = '../../data/stanfordSentimentTreebank'
    sentence_file = 'datasetSentences.txt'
    label_file = 'sentiment_labels.txt'
    split_file = 'datasetSplit.txt'

    sentence_df = pd.read_csv(os.path.join(data_dir, sentence_file), sep='\t')
    label_df = pd.read_csv(os.path.join(data_dir, label_file), sep='|', usecols=['sentiment values'])
    split_df = pd.read_csv(os.path.join(data_dir, split_file), sep=',', usecols=['splitset_label'])
    label_df['label'] = label_df['sentiment values'].apply(score2label)

    df = pd.concat([sentence_df, label_df['label'], split_df], axis=1)
    train_df = df[df['splitset_label'] == 1]
    validate_df = df[df['splitset_label'] == 3]
    test_df = df[df['splitset_label'] == 2]
    assert len(train_df) == 8544
    assert len(validate_df) == 1101
    assert len(test_df) == 2210
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    validate_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
