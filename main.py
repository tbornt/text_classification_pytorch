import os
import time
import errno
import pickle
import argparse
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import check_fields, print_progress, AverageMeter, accuracy, save_checkpoint
from utils.dataloader import load_data
from models.classifier import RNNTextClassifier


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--config', required=True, help='config file')
config_parser = configparser.ConfigParser()


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    i = 0
    for train_item in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        text, lengths = train_item.comment_text
        text.transpose_(0, 1)
        label = train_item.toxic
        sort_idx = np.argsort(-lengths)

        text = text[sort_idx, :]
        lengths = lengths[sort_idx]
        label = label[sort_idx]

        if torch.cuda.is_available():
            text = text.cuda()
            label = label.cuda()

        # compute output
        output = model(text, lengths)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1 = accuracy(output, label, topk=(1,))
        losses.update(loss.item(), text.size(0))
        top1.update(acc1[0].item(), text.size(0))

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
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
        i += 1


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        i = 0
        for test_item in val_loader:
            text, lengths = test_item.comment_text
            text.transpose_(0, 1)
            label = test_item.toxic
            sort_idx = np.argsort(-lengths)

            text = text[sort_idx, :]
            lengths = lengths[sort_idx]
            label = label[sort_idx]

            if torch.cuda.is_available():
                text = text.cuda()
                label = label.cuda()

            # compute output
            output = model(text, lengths)
            loss = criterion(output, label)

            # measure accuracy and record loss
            acc1 = accuracy(output, label, topk=(1, ))
            losses.update(loss.item(), text.size(0))
            top1.update(acc1[0].item(), text.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
            i += 1
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


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
                    required_fields = ['train_file', 'text_column', 'label_column', 'batch_size']
                    check_fields(required_fields, IO_session)
                    train_iter, test_iter, TEXT = load_data(file_type, IO_session, is_train=is_train)
                    vocab = TEXT.vocab
        else:
            raise Exception('file format should be configured in IO session')
        output_dir = IO_session.get('output_dir', 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pickle.dump(vocab, open('vocab.cache', 'wb'))
        # print('%d train samples and %d test samples' % (len(train_dataset), len(test_dataset)))
        print_progress("IO config Done")
    else:
        raise Exception('IO should be configured in config file')

    if 'MODEL' in sessions:
        print_progress("Start config Model")
        MODEL_session = config_parser['MODEL']
        required_fields = ['type', 'rnn_type', 'embedding_size', 'hidden_size', 'n_label']
        check_fields(required_fields, MODEL_session)
        for key, val in MODEL_session.items():
            print(key, '=', val)
        clf_type = MODEL_session['type']
        if clf_type.lower() == 'rnn':
            model = RNNTextClassifier(vocab, MODEL_session)
        if torch.cuda.is_available():
            model = model.cuda()
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
            optimizer = optim.Adam(model.parameters(), lr=lr)
            if torch.cuda.is_available():
                criterion = nn.CrossEntropyLoss().cuda()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            raise Exception('TRAIN should be configured in config file')
    else:
        if 'DECODE' in sessions:
            print_progress('Start DECODE config')
        else:
            raise Exception('DECODE should be configured in config file')

    best_acc1 = 0
    if is_train:
        for epoch in range(n_epoch):
            train(train_iter, model, criterion, optimizer, epoch, record_step)

            acc1 = validate(test_iter, model, criterion, record_step)
            is_best = acc1 > best_acc1

            best_acc1 = max(acc1, best_acc1)
            save_checkpoint(output_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict()
            }, is_best, output_dir)
