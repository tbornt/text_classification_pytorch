import os
import time
import errno
import pickle
import argparse
import configparser

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from sklearn.utils.extmath import softmax
from utils.utils import check_fields, print_progress, AverageMeter, class_eval, save_checkpoint, \
    load_checkpoint
from utils.utils import accuracy as acc_func, str2list
from utils.focalloss import FocalLoss
from utils.dataloader import load_data
from models.rnn_classifier import RNNTextClassifier
from models.cnn_classifier import CNNTextClassifier
from models.dpcnn_classifier import DPCNNTextClassifier
from models.rcnn_classifier import RCNNTextClassifier
from models.multiheadattention_classifier import MultiHeadAttention_classifier


import warnings
warnings.filterwarnings('ignore')


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--config', required=True, help='config file')
config_parser = configparser.ConfigParser()


def train(train_loader, model, loss_type, criterion, optimizer, epoch, print_freq, text_column, label_column, n_label, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    i = 0
    for train_item in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        text, lengths = getattr(train_item, text_column)
        text.transpose_(0, 1)  # batch first
        sort_idx = np.argsort(-lengths)
        text = text[sort_idx, :]
        lengths = lengths[sort_idx]

        if loss_type.lower() in ['focal', 'cross_entropy']:
            if len(label_column) != 1:
                raise Exception("only one label needed for %s loss" % loss_type.lower())
            label = getattr(train_item, label_column[0])
            labels = label[sort_idx]
        elif loss_type.lower() in ['binary_cross_entropy']:
            labels = []
            for column in label_column:
                label = getattr(train_item, column)
                label = label[sort_idx]
                label = label.unsqueeze(1)
                labels.append(label)
            labels = torch.cat(labels, 1).float()
        else:
            raise Exception('%s loss is not supported' % loss_type.lower())

        if torch.cuda.is_available():
            text = text.cuda()
            labels = labels.cuda()

        # compute output
        output = model(text, lengths)
        loss = criterion(output, labels)

        if loss_type.lower() in ['focal', 'cross_entropy']:
            output = F.softmax(output)
            pred_type = 'multi_class'
        if loss_type.lower() in ['binary_cross_entropy']:
            output = F.sigmoid(output)
            pred_type= 'multi_label'

        # measure accuracy and record loss
        acc, precision, recall, fscore, auc_score = \
            class_eval(output.data.cpu(), labels, pred_type)
        accuracies.update(acc, labels.size(0))
        precisions.update(precision, labels.size(0))
        recalls.update(recall, labels.size(0))
        fscores.update(fscore, labels.size(0))
        auc_scores.update(auc_score, labels.size(0))

        losses.update(loss.item(), text.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses,
                                                             top1=accuracies,
                                                             prec=precisions,
                                                             recall=recalls,
                                                             f1=fscores,
                                                             auc=auc_scores))

        i += 1
    writer.add_scalar('Train Loss', losses.avg, epoch)


def validate(val_loader, model, criterion, print_freq, text_column, label_column, n_label, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        i = 0
        for test_item in val_loader:
            text, lengths = getattr(test_item, text_column)
            text.transpose_(0, 1)  # batch first
            sort_idx = np.argsort(-lengths)
            text = text[sort_idx, :]
            lengths = lengths[sort_idx]

            if loss_type.lower() in ['focal', 'cross_entropy']:
                if len(label_column) != 1:
                    raise Exception("only one label needed for %s loss" % loss_type.lower())
                label = getattr(test_item, label_column[0])
                labels = label[sort_idx]
            elif loss_type.lower() in ['binary_cross_entropy']:
                labels = []
                for column in label_column:
                    label = getattr(test_item, column)
                    label = label[sort_idx]
                    label = label.unsqueeze(1)
                    labels.append(label)
                labels = torch.cat(labels, 1).float()
            else:
                raise Exception('%s loss is not supported' % loss_type.lower())

            if torch.cuda.is_available():
                text = text.cuda()
                labels = labels.cuda()

            # compute output
            output = model(text, lengths)
            loss = criterion(output, labels)

            if loss_type.lower() in ['focal', 'cross_entropy']:
                output = F.softmax(output)
                pred_type = 'multi_class'
            if loss_type.lower() in ['binary_cross_entropy']:
                output = F.sigmoid(output)
                pred_type = 'multi_label'

            # measure accuracy and record loss
            acc, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), labels, pred_type)
            accuracies.update(acc, labels.size(0))
            precisions.update(precision, labels.size(0))
            recalls.update(recall, labels.size(0))
            fscores.update(fscore, labels.size(0))
            auc_scores.update(auc_score, labels.size(0))

            losses.update(loss.item(), text.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(epoch, i, len(val_loader),
                                                                 batch_time=batch_time,
                                                                 loss=losses,
                                                                 top1=accuracies,
                                                                 prec=precisions,
                                                                 recall=recalls,
                                                                 f1=fscores,
                                                                 auc=auc_scores))
            i += 1

        print(' * Acc {top1.avg:.3f}'.format(top1=accuracies))
        print(' * Precision {top1.avg:.3f}'.format(top1=precisions))
        print(' * Recall {top1.avg:.3f}'.format(top1=recalls))
        print(' * F1 {top1.avg:.3f}'.format(top1=fscores))
        print(' * AUC {top1.avg:.3f}'.format(top1=auc_scores))
    writer.add_scalar('Validate Loss', losses.avg, epoch)
    writer.add_scalar('Validate Acc', accuracies.avg, epoch)
    return accuracies.avg, precisions.avg, recalls.avg, fscores.avg, auc_scores.avg


def decode(decode_iter, model, output_file, text_column, output_type, task_type):
    model.eval()
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = open(output_file, 'w')
    with torch.no_grad():
        for decode_item in decode_iter:
            text, lengths = getattr(decode_item, text_column)
            text.transpose_(0, 1)
            sort_idx = np.argsort(-lengths)
            unsort_idx = np.argsort(sort_idx)

            text = text[sort_idx, :]
            lengths = lengths[sort_idx]

            if torch.cuda.is_available():
                text = text.cuda()

            output = model(text, lengths)
            if task_type == 'multi_class':
                if output_type == 'label':
                    _, pred = output.max(1)
                    # return to origin order
                    pred = pred[unsort_idx]
                    for label in pred:
                        output_file.write('%d\n' % label.item())
                elif output_type == 'prob':
                    output = F.softmax(output)
                    output = output[unsort_idx]
                    for prob in output:
                        prob = [str(p) for p in prob]
                        output_file.write(','.join(prob)+'\n')
            elif task_type == 'multi_label':
                if output_type == 'label':
                    pass
                elif output_type == 'prob':
                    output = F.sigmoid(output)
                    output = output[unsort_idx]
                    for prob in output:
                        prob = [str(p.item()) for p in prob]
                        output_file.write(','.join(prob)+'\n')
            else:
                raise Exception('task type % s not allowed' % task_type)
    output_file.close()

if __name__ == '__main__':
    args = arg_parser.parse_args()
    config_file = args.config
    if not os.path.exists(config_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    config_parser.read(config_file)
    sessions = config_parser.sections()
    if 'STATUS' in sessions:
        is_train = config_parser['STATUS'].get('status', 'train') == 'train'
        if is_train:
            print_progress('Start Training')
        else:
            print_progress('Start Decoding')
    else:
        is_train = True

    if 'IO' in sessions:
        print_progress("Start config IO")
        IO_session = config_parser['IO']
        for key, val in IO_session.items():
            print(key, '=', val)
        file_type = IO_session.get('type', None)
        if file_type:
            if file_type == 'csv':
                if is_train:
                    # training
                    required_fields = ['train_file', 'text_column', 'label_column', 'batch_size']
                    check_fields(required_fields, IO_session)
                    text_column = IO_session['text_column']
                    label_column = str2list(IO_session['label_column'])
                    train_iter, test_iter, TEXT = load_data(file_type, IO_session, is_train=is_train)
                    vocab = TEXT.vocab
                    # save vocab
                    output_dir = IO_session.get('output_dir', 'output')
                    writer = SummaryWriter(output_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    pickle.dump(vocab, open(os.path.join(output_dir, 'vocab.cache'), 'wb'))
                else:
                    # decoding
                    required_fields = ['decode_file', 'text_column', 'vocab_file', 'batch_size']
                    check_fields(required_fields, IO_session)
                    text_column = IO_session['text_column']
                    output_file = IO_session.get('output_file', 'output/output.csv')
                    decode_iter, TEXT = load_data(file_type, IO_session, is_train=is_train)
                    vocab = TEXT.vocab
        else:
            raise Exception('file format should be configured in IO session')
        # print('%d train samples and %d test samples' % (len(train_dataset), len(test_dataset)))
        print_progress("IO config Done")
    else:
        raise Exception('IO should be configured in config file')

    if 'MODEL' in sessions:
        print_progress("Start config Model")
        MODEL_session = config_parser['MODEL']
        required_fields = ['type']
        check_fields(required_fields, MODEL_session)
        clf_type = MODEL_session['type']
        if clf_type.lower() == 'rnn':
            required_fields = ['rnn_type', 'embedding_size', 'hidden_size', 'n_label']
            check_fields(required_fields, MODEL_session)
            n_label = int(MODEL_session['n_label'])  # required for later focal loss init
            for key, val in MODEL_session.items():
                print(key, '=', val)
            model = RNNTextClassifier(vocab, MODEL_session)
        elif clf_type.lower() == 'textcnn':
            required_fields = ['embedding_size', 'n_label']
            check_fields(required_fields, MODEL_session)
            n_label = int(MODEL_session['n_label'])  # required for later focal loss init
            for key, val in MODEL_session.items():
                print(key, '=', val)
            model = CNNTextClassifier(vocab, MODEL_session)
        elif clf_type.lower() == 'dpcnn':
            required_fields = ['embedding_size', 'n_label']
            check_fields(required_fields, MODEL_session)
            n_label = int(MODEL_session['n_label'])  # required for later focal loss init
            for key, val in MODEL_session.items():
                print(key, '=', val)
            model = DPCNNTextClassifier(vocab, MODEL_session)
        elif clf_type.lower() == 'rcnn':
            required_fields = ['rnn_type', 'embedding_size', 'hidden_size', 'n_label']
            check_fields(required_fields, MODEL_session)
            n_label = int(MODEL_session['n_label'])  # required for later focal loss init
            for key, val in MODEL_session.items():
                print(key, '=', val)
            model = RCNNTextClassifier(vocab, MODEL_session)
        elif clf_type.lower() == 'transformer':
            required_fields = ['n_label']
            check_fields(required_fields, MODEL_session)
            n_label = int(MODEL_session['n_label'])  # required for later focal loss init
            required_fields = ['fix_length']
            check_fields(required_fields, IO_session)
            fix_len = IO_session['fix_length']
            MODEL_session['fix_length'] = fix_len
            for key, val in MODEL_session.items():
                print(key, '=', val)
            model = MultiHeadAttention_classifier(vocab, MODEL_session)
        else:
            raise Exception('Other Model is not supported')
        if torch.cuda.is_available():
            model = model.cuda()
        if not is_train:
            required_fields = ['checkpoint']
            check_fields(required_fields, MODEL_session)
            load_checkpoint(model, MODEL_session['checkpoint'])
        print(model)
        print_progress("Model config Done")
    else:
        raise Exception('MODEL should be configured in config file')

    if is_train:
        if 'TRAIN' in sessions:
            print_progress('Start TRAIN coinfg')
            TRAIN_session = config_parser['TRAIN']
            for key, val in TRAIN_session.items():
                print(key, '=', val)
            required_fields = ['n_epoch', 'use_gpu', 'learning_rate']
            check_fields(required_fields, TRAIN_session)
            n_epoch = int(TRAIN_session['n_epoch'])
            lr = float(TRAIN_session['learning_rate'])
            record_step = int(TRAIN_session.get('record_step', 100))
            optim_type = TRAIN_session.get('optim_type', 'adam')
            weigth_decay = float(TRAIN_session.get('weight_decay', 0.0))
            loss_type = TRAIN_session.get('loss_type', 'cross_entropy')
            if optim_type.lower() == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weigth_decay)
            elif optim_type.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=lr, weigth_decay=weigth_decay)
            else:
                raise Exception('Other optimizer is not supported')

            if loss_type.lower() == 'cross_entropy':
                if torch.cuda.is_available():
                    criterion = nn.CrossEntropyLoss().cuda()
                else:
                    criterion = nn.CrossEntropyLoss()
            elif loss_type.lower() == 'focal':
                if torch.cuda.is_available():
                    criterion = FocalLoss(n_label).cuda()
                else:
                    criterion = FocalLoss(n_label)
            elif loss_type.lower() == 'binary_cross_entropy':
                if torch.cuda.is_available():
                    criterion = nn.BCEWithLogitsLoss().cuda()
                else:
                    criterion = nn.BCEWithLogitsLoss()
            else:
                raise Exception('%s loss is not supported' % loss_type.lower())
            print_progress('TRAIN coinfg Done')
        else:
            raise Exception('TRAIN should be configured in config file')
    else:
        if 'DECODE' in sessions:
            print_progress('Start DECODE config')
            DECODE_session = config_parser['DECODE']
            required_fields = ['use_gpu']
            output_type = DECODE_session.get('output_type', 'label')
            task_type = DECODE_session.get('task_type', 'multi_class')
            print_progress('DECODE config Done')
        else:
            raise Exception('DECODE should be configured in config file')

    if is_train:
        best_acc1 = 0
        best_precision = 0
        best_recall = 0
        best_f1 = 0
        best_auc = 0
        for epoch in range(n_epoch):
            train(train_iter,
                  model,
                  loss_type,
                  criterion,
                  optimizer,
                  epoch,
                  record_step,
                  text_column,
                  label_column,
                  n_label,
                  writer)

            acc1, precision, recall, f1, auc = validate(test_iter,
                                                        model,
                                                        criterion,
                                                        record_step,
                                                        text_column,
                                                        label_column,
                                                        n_label,
                                                        writer)
            if n_label == 2:
                is_best = auc > best_auc
            else:
                is_best = acc1 > best_acc1

            best_acc1 = max(acc1, best_acc1)
            best_precision = max(precision, best_precision)
            best_recall = max(recall, best_recall)
            best_f1 = max(f1, best_f1)
            best_auc = max(auc, best_auc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict()
            }, is_best, output_dir)
    else:
        decode(decode_iter, model, output_file, text_column, output_type, task_type)
