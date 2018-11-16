import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


result_dir = 'result'
result_file = 'bigru.csv'

truth_file = '../../data/stanfordSentimentTreebank/test.csv'

if __name__ == '__main__':
    truth = pd.read_csv(truth_file, usecols=['label'])
    truth = truth['label'].values
    pred_file = os.path.join(result_dir, result_file)
    pred = pd.read_csv(pred_file, names=['pred_label'], header=None)
    pred = pred['pred_label'].values
    res = confusion_matrix(truth, pred)
    print(res)
    acc_score = accuracy_score(truth, pred)
    print(acc_score)
