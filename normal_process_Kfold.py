import argparse
import os
import shutil
import sys
import time
import pandas as pd
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler

from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                               'classification task (default: regression)')
parser.add_argument('--datatypes', '--dt', choices=['xenonpy', 'init'],
                    default='xenonpy', help='complete a different atom feature descriptors')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                         help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default '
                              '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('-b', '--batch-size', default=36, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[250], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--gamma', default=0.1, type=float, metavar='M',
                    help='gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 0)')
args = parser.parse_args(sys.argv[1:])

if args.datatypes == 'xenonpy':
    from cgcnn.data import CIFData
    from cgcnn.data import collate_pool, get_train_val_test_loader
elif args.datatypes == 'init':
    from cgcnn.data_init import CIFData
    from cgcnn.data_init import collate_pool, get_train_val_test_loader

if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()


def K_fold(dataset, cv=5, collate_fn=collate_pool, batch_size=args.batch_size,
           num_workers=0, pin_memory=False):
    total_size = len(dataset)
    test_ratio = float(1 / cv)
    test_size = int(test_ratio * total_size)
    indices = list(range(total_size))
    random.shuffle(indices)
    if total_size % cv == 0:
        step = int(total_size / cv)
    else:
        step = int(total_size / cv) + 1
    train_data = []
    test_data = []
    for i in range(cv):
        test_sampler = SubsetRandomSampler(indices[i * step:i * step + step])
        train_sampler = SubsetRandomSampler(indices[0:i * step] + indices[i * step + step:total_size])
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory)
        train_data.append(train_loader)
        test_data.append(test_loader)
    return train_data, test_data


def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    train_data, test_data = K_fold(dataset)
    with open('test_results_TL.csv', 'w') as f:
        f.truncate()
        f.close()
    for cv in range(5):
        if args.task == 'regression':
            best_mae_error = 1e10
        else:
            best_mae_error = 0.
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                                           'classification' else False)
        if args.cuda:
            model.cuda()
        # define loss func and optimizer
        if model_args.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()
        normalizer = Normalizer(torch.zeros(3))
        # optionally resume from a checkpoint
        if os.path.isfile(args.modelpath):
            print("=> loading model '{}'".format(args.modelpath))
            checkpoint = torch.load(args.modelpath,
                                    map_location=lambda storage, loc: storage)

            '''
            Transfer model_state_dict to new model
            '''
            TL_state_dict = checkpoint['state_dict']
            state_dict = model.state_dict().copy()

            state_dict['Normalize.weight'] = TL_state_dict['Normalize.weight']
            state_dict['Normalize.bias'] = TL_state_dict['Normalize.bias']
            state_dict['Normalize.running_mean'] = TL_state_dict['Normalize.running_mean']
            state_dict['Normalize.running_var'] = TL_state_dict['Normalize.running_var']
            state_dict['Normalize.num_batches_tracked'] = TL_state_dict['Normalize.num_batches_tracked']
            state_dict['embedding.weight'] = TL_state_dict['embedding.weight']
            state_dict['embedding.bias'] = TL_state_dict['embedding.bias']

            state_dict['convs.0.fc_full.weight'] = TL_state_dict['convs.0.fc_full.weight']
            state_dict['convs.0.fc_full.bias'] = TL_state_dict['convs.0.fc_full.bias']
            state_dict['convs.0.bn1.weight'] = TL_state_dict['convs.0.bn1.weight']
            state_dict['convs.0.bn1.bias'] = TL_state_dict['convs.0.bn1.bias']
            state_dict['convs.0.bn1.running_mean'] = TL_state_dict['convs.0.bn1.running_mean']
            state_dict['convs.0.bn1.running_var'] = TL_state_dict['convs.0.bn1.running_var']
            state_dict['convs.0.bn1.num_batches_tracked'] = TL_state_dict['convs.0.bn1.num_batches_tracked']
            state_dict['convs.0.bn2.weight'] = TL_state_dict['convs.0.bn2.weight']
            state_dict['convs.0.bn2.bias'] = TL_state_dict['convs.0.bn2.bias']
            state_dict['convs.0.bn2.running_mean'] = TL_state_dict['convs.0.bn2.running_mean']
            state_dict['convs.0.bn2.running_var'] = TL_state_dict['convs.0.bn2.running_var']
            state_dict['convs.0.bn2.num_batches_tracked'] = TL_state_dict['convs.0.bn2.num_batches_tracked']
            '''
            state_dict['convs.1.fc_full.weight'] = TL_state_dict['convs.1.fc_full.weight']
            state_dict['convs.1.fc_full.bias'] = TL_state_dict['convs.1.fc_full.bias']
            state_dict['convs.1.bn1.weight'] = TL_state_dict['convs.1.bn1.weight']
            state_dict['convs.1.bn1.bias'] = TL_state_dict['convs.1.bn1.bias']
            state_dict['convs.1.bn1.running_mean'] = TL_state_dict['convs.1.bn1.running_mean']
            state_dict['convs.1.bn1.running_var'] = TL_state_dict['convs.1.bn1.running_var']
            state_dict['convs.1.bn1.num_batches_tracked'] = TL_state_dict['convs.1.bn1.num_batches_tracked']
            state_dict['convs.1.bn2.weight'] = TL_state_dict['convs.1.bn2.weight']
            state_dict['convs.1.bn2.bias'] = TL_state_dict['convs.1.bn2.bias']
            state_dict['convs.1.bn2.running_mean'] = TL_state_dict['convs.1.bn2.running_mean']
            state_dict['convs.1.bn2.running_var'] = TL_state_dict['convs.1.bn2.running_var']
            state_dict['convs.1.bn2.num_batches_tracked'] = TL_state_dict['convs.1.bn2.num_batches_tracked']
            
            state_dict['conv_to_fc.weight'] = TL_state_dict['conv_to_fc.weight']
            state_dict['conv_to_fc.bias'] = TL_state_dict['conv_to_fc.bias']
            
            state_dict['fc_out.weight'] = TL_state_dict['fc_out.weight']
            state_dict['fc_out.bias'] = TL_state_dict['fc_out.bias']
            '''
            model.load_state_dict(state_dict)
            print('state_dict loaded successful')
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded model '{}' (epoch {}, validation {})"
                  .format(args.modelpath, checkpoint['epoch'],
                          checkpoint['best_mae_error']))
        else:
            print("=> no model found at '{}'".format(args.modelpath))

        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                   weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_mae_error = checkpoint['best_mae_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                normalizer.load_state_dict(checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=args.gamma)

        train_loader = train_data[cv]
        test_loader = test_data[cv]
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, normalizer)

            # evaluate on validation set
            mae_error, R2 = validate(test_loader, model, criterion, normalizer)
            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)

            scheduler.step()

            # remember the best mae_eror and save checkpoint
            if args.task == 'regression':
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict().copy(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict().copy(),
                'normalizer': normalizer.state_dict().copy(),
                'args': vars(args)
            }, is_best)

        # test best model
        print('---------Evaluate Model on Test Set---------------')
        best_checkpoint = torch.load('model_TL.pth.tar')
        model.load_state_dict(best_checkpoint['state_dict'])
        validate(test_loader, model, criterion, normalizer, test=True)
    print('**********Evaluate Model on All Datasets**********')
    result = pd.read_csv('test_results_TL.csv', header=None)
    mae_all = mean_absolute_error(np.array(result[1]), np.array(result[2]))
    r2_all = r2_score(np.array(result[1]), np.array(result[2]))
    print('(`A`)!!!', 'mae:', mae_all)
    print('r2:', r2_all)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        LR = optimizer.param_groups[0]['lr']
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                      'lr {LR}'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, mae_errors=mae_errors, LR=LR)
                      )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            R2 = r2_score(target.view(-1).tolist(), normalizer.denorm(output.data.cpu()).view(-1).tolist())
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
              'R\u00b2 {R2:.3f}'.format(i, len(val_loader), batch_time=batch_time, loss=losses,
                                        mae_errors=mae_errors, R2=R2))

    if test:
        star_label = '**'
        import csv
        with open('test_results_TL.csv', 'a') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        with open('results_normal_ph.csv', 'a') as f1:
            writer = csv.writer(f1)
            writer.writerow(["r2_score: %3f" % R2])
            writer.writerow(['mae: %3f' % mae_errors.avg])
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors:.3f}\t '
          'R2 {R2:.3f}'.format(star=star_label, mae_errors=mae_errors.avg, R2=R2))
    return mae_errors.avg, R2


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_TL.pth.tar')


if __name__ == '__main__':
    main()
