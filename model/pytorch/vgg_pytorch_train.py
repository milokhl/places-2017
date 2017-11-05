import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import vgg_pytorch as VGG

from miniplaces_dataset import *

import time
import shutil

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def main():
    # Apply a series of transformations to the input data.
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.45834960097, 0.44674252445, 0.41352266842), (0.5, 0.5, 0.5))]
    ) # TODO: change the STD vals

    # Load in the training set.
    training_set = MiniPlacesDataset('/home/milo/envs/tensorflow35/miniplaces/data/train.txt',
                                     '/home/milo/envs/tensorflow35/miniplaces/data/images/',
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=True, num_workers=10)

    val_set = MiniPlacesDataset('/home/milo/envs/tensorflow35/miniplaces/data/val.txt',
                                '/home/milo/envs/tensorflow35/miniplaces/data/images/',
                                transform=transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.45834960097, 0.44674252445, 0.41352266842), (0.5, 0.5, 0.5))]))

    val_loader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=True, num_workers=10)

    # Define the model, loss, and optimizer.
    model = VGG.vgg11(num_classes=100)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    # Parameters
    epochs = 10
    print_freq = 20

    for epoch in range(epochs):

        # Train for one epoch.
        train(train_loader, model, criterion, optimizer, epoch, print_freq=print_freq)

        # Every epoch, test on the validation data.
        prec1 = validate(val_loader, model, criterion, print_freq=print_freq)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # Save a checkpoint file.
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'vgg',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    is_best = True

    end = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - end)

        # Get input and label tensors, wrap them in variables.
        inputs, target = data
        target = target.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs)
        targets_var = torch.autograd.Variable(target)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward propagation, and optimization.
        outputs = model(inputs_var)
        loss = criterion(outputs, targets_var)
        loss.backward()
        optimizer.step()

        # Update metrics.
        prec1, prec5 = accuracy(outputs, targets_var, topk=(1, 5))
        losses.update(loss.data[0], inputs_var.size(0))
        top1.update(prec1.data[0], inputs_var.size(0))
        top5.update(prec5.data[0], inputs_var.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Print out metrics periodically.
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('Finished Training epoch!')

def validate(val_loader, model, criterion, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()