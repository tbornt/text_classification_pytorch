import os
import shutil

import torch


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
