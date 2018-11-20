import os
import shutil

import torch
import numpy as np
from sklearn import metrics


def get_class(x):
    res = []
    for row in x:
        row_res = []
        for col in row:
            if col >= 0.5:
                row_res.append(1)
            else:
                row_res.append(0)
        res.append(row_res)
    return np.array(res)


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def class_eval(prediction, target, pred_type):
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    if prediction.shape[1] == 2:
        pred_label = np.argmax(prediction, axis=1)
        target_label = np.squeeze(target)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        try:
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        except:  # all true label are 0
            auc_score = 0.0
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        if pred_type == 'multi_label':
            get_class_vec = np.vectorize(get_class)
            pred_label = get_class(prediction)
            accuracy = hamming_score(target, pred_label)
            precision = 0.0
            recall = 0.0
            fscore = 0.0
            auc_score = 0.0
        elif pred_type == 'multi_class':
            pred_label = np.argmax(prediction, axis=1)
            target_label = np.argmax(prediction, axis=1)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                target_label, pred_label, average='binary')
            auc_score = 0.0
            accuracy = metrics.accuracy_score(target_label, pred_label)
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        best_model = os.path.join(path, 'model_best.pth.tar')
        shutil.copyfile(filename, best_model)


def load_checkpoint(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))


def check_fields(required_fields, session):
    for field in required_fields:
        if field not in session:
            raise Exception('%s should be configured in IO session' % field)


def print_progress(text):
    print("=====%s=====" % text)


def str2list(str, sep=','):
    return str.split(sep)
