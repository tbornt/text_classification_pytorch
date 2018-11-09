import os
import errno
import argparse
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import check_fields, print_progress
from utils.dataloader import load_data
from models.classifier import RNNTextClassifier


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--config', required=True, help='config file')
config_parser = configparser.ConfigParser()


def train(model, text, label, loss_func, opt, running_loss):
    opt.zero_grad()
    output = model(text, lengths)
    loss = loss_func(output, label)
    batch_loss = loss.data.item()
    running_loss += batch_loss
    loss.backward()
    opt.step()
    return batch_loss, running_loss


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
                    train_iter, test_iter, vocab = load_data(file_type, IO_session, is_train=is_train)
        else:
            raise Exception('file format should be configured in IO session')
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
            opt = optim.Adam(model.parameters(), lr=lr)
            loss_func = nn.CrossEntropyLoss()
        else:
            raise Exception('TRAIN should be configured in config file')
    else:
        if 'DECODE' in sessions:
            print_progress('Start DECODE config')
        else:
            raise Exception('DECODE should be configured in config file')

    if is_train:
        for epoch in range(n_epoch):
            model.train()
            running_loss = 0.0
            i = 0
            for train_item in train_iter:
                i += 1
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

                batch_loss, running_loss = train(model, text, label, loss_func, opt, running_loss)
                if i % record_step == 0:
                    print('current batch loss: %f' % batch_loss)
            print('epoch%d loss: %f' % (epoch, running_loss/i))

            correct = 0
            total = 0
            for test_item in test_iter:
                model.eval()

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

                output = model(text, lengths)
                _, pred_label = torch.topk(output, 1)
                label = label.view(len(label), -1)
                correct += (pred_label == label).sum()
                total += output.shape[0]
            print(correct.item(), total)

