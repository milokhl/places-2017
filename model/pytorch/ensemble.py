# Submission format: <filename> <label(1)> <label(2)> <label(3)> <label(4)> <label(5)>
# Example line: val/00000005.jpg 65 3 84 93 67

# Your team, LaCroixNet, has been registered for the challenge.
# Your team code (case-sensitive) for submitting to the leaderboard is: fFR7jWuG2XqiAImnJQrN
from __future__ import print_function, division
import os, sys, time

sys.path.append('./convnet')
sys.path.append('./capsnet')

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import vgg_pytorch as VGG
from miniplaces_capsnet3 import PlacesCapsuleNet

from miniplaces_dataset import *
from utils import accuracy, AverageMeter, save_checkpoint, log

BATCH_SIZE = 1
VGG_LOAD_EPOCH = './final/model_best.pth.tar'
CAPSNET_LOAD_EPOCH = './final/epoch_16.pt'

# Apply a series of transformations to the input data.
DATA_MEAN = (0.45834960097, 0.44674252445, 0.41352266842)
DATA_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 128

# Weights for ensembling.
MODEL1_WEIGHT = 0.7
MODEL2_WEIGHT = 0.3
assert(MODEL1_WEIGHT + MODEL2_WEIGHT == 1.0)


def validate(val_loader, models, print_freq=1):
  for model in models:
    model.eval()

  print('Starting validation of ensemble!')
  model1_weight = torch.Tensor([MODEL1_WEIGHT]).cuda()
  model2_weight = torch.Tensor([MODEL2_WEIGHT]).cuda()

  batch_time = AverageMeter()
  top1_ensemble = AverageMeter()
  top5_ensemble = AverageMeter()

  top1_model1 = AverageMeter()
  top1_model2 = AverageMeter()

  top5_model1 = AverageMeter()
  top5_model2 = AverageMeter()

  end = time.time()
  for i, (input, target) in enumerate(val_loader):
    print(input.size())
    target = target.cuda(async=True)
    input_var = Variable(input, volatile=True).cuda()
    target_var = Variable(target, volatile=True)

    # Get outputs from each model in ensemble.
    outputs = [model(input_var) for model in models]

    # Weight each model and combine outputs.
    ensemble_output = model1_weight * outputs[0].data + model2_weight * outputs[1][0].data # Add extra index to caps net because of reconstruction

    # Measure accuracies.
    prec1 = [accuracy(outputs[0].data, target, topk=(1,)), accuracy(outputs[1][0].data, target, topk=(1,))]
    prec5 = [accuracy(outputs[0].data, target, topk=(5,)), accuracy(outputs[1][0].data, target, topk=(5,))]
    top1_model1.update(prec1[0][0], input.size(0))
    top1_model2.update(prec1[1][0], input.size(0))
    top5_model1.update(prec5[0][0], input.size(0))
    top5_model2.update(prec5[1][0], input.size(0))

    # Get the enemble accuracies.
    prec1_ensemble, prec5_ensemble = accuracy(ensemble_output, target, topk=(1,5))
    top1_ensemble.update(prec1_ensemble, input.size(0))
    top5_ensemble.update(prec5_ensemble, input.size(0))

    # Measure time.
    batch_time.update(time.time() - end)
    end = time.time()

    print_str = 'Validation: [%d/%d] \t Prec1(ens): %f (%f) Prec5(ens): %f (%f) \n' \
                % (i, len(val_loader), top1_ensemble.val[0], top1_ensemble.avg[0], top5_ensemble.val[0], top5_ensemble.avg[0])
    print_str += 'Prec1(m1): %f (%f) Prec5(m1): %f (%f) \t Prec1(m2): %f (%f) Prec5(m2): %f (%f)'\
                % (top1_model1.val[0], top1_model1.avg[0], top5_model1.val[0], top5_model1.avg[0],
                  top1_model2.val[0], top1_model2.avg[0], top5_model2.val[0], top5_model2.avg[0])

    # Print statistacks.
    if i % print_freq == 0:
      print(print_str)

  # Print out final statistacks.
  print_str = 'Validation: [%d/%d] \t Prec1(ens): %f (%f) Prec5(ens): %f (%f) \n' \
              % (i, len(val_loader), top1_ensemble.val[0], top1_ensemble.avg[0], top5_ensemble.val[0], top5_ensemble.avg[0])
  print_str += 'Prec1(m1): %f (%f) Prec5(m1): %f (%f) Prec1(m2): %f (%f) Prec5(m2): %f (%f)'\
              % (top1_model1.val[0], top1_model1.avg[0], top5_model1.val[0], top5_model1.avg[0],
                top1_model2.val[0], top1_model2.avg[0], top5_model2.val[0], top5_model2.avg[0])
  print(print_str)

  print('Finished validation!')
  return top1_ensemble.avg, top5_ensemble.avg


def generate_submission_ensemble(models, transforms_list, print_freq=100):

  # Set up a test loader, which outputs image / filename pairs.
  test_set = MiniPlacesTestSet('../../data/images/test/',
                               transform=transforms.Compose(transforms_list),
                               outfile=str(int(time.time())) + 'predictions.txt')
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

  # Switch models to eval mode.
  for model in models:
    model.eval()

  print('Generating submission!')
  model1_weight = torch.Tensor([MODEL1_WEIGHT]).cuda()
  model2_weight = torch.Tensor([MODEL2_WEIGHT]).cuda()

  for i, data in enumerate(test_loader):
      image, filename = data
      print(image.size())
      # image = image.unsqueeze(0)
      print(image.size())
      input_var = torch.autograd.Variable(image).cuda()

      # Get outputs from each model in ensemble.
      outputs = [model(input_var) for model in models]

      # Weight each model and combine outputs.
      # Add extra index to caps net because of reconstruction
      ensemble_output = model1_weight * outputs[0].data + model2_weight * outputs[1][0].data

      _, top5 = ensemble_output.topk(5, 1, True, True)
      top5 = top5.t()

      labels = [top5[i][0] for i in range(5)]

      # Write the top 5 labels as a new line.
      test_set.write_labels(filename, labels)

      # Print statistacks.
      if i % print_freq == 0:
        print('Finished %d/%d' % (i, len(test_loader)))

  print('Finished generating submission!')

def generate_submission(model, transforms_list, print_freq=100):
  # Set up a test loader, which outputs image / filename pairs.
  test_set = MiniPlacesTestSet('../../data/images/test/',
                               transform=transforms.Compose(transforms_list),
                               outfile=str(int(time.time())) + 'predictions.txt')
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

  model.eval()

  print('Generating submission!')
  for i, data in enumerate(test_loader):
      image, filename = data
      input_var = torch.autograd.Variable(image).cuda()

      # Get outputs from each model in ensemble.
      output = model(input_var)
      _, top5 = output.topk(5, 1, True, True)
      top5 = top5.t()
      labels = [top5.data[i][0] for i in range(5)]

      # Write the top 5 labels as a new line.
      test_set.write_labels('test/' + filename[0], labels)

      # Print statistacks.
      if i % print_freq == 0:
        print('Finished %d/%d' % (i, len(test_loader)))

  print('Finished generating submission!')

if __name__ == '__main__':

  # Set up VGG16
  model_vgg16 = VGG.vgg16(num_classes=100, dropout=0.5, light=True)
  model_vgg16.features = torch.nn.DataParallel(model_vgg16.features)
  model_vgg16.cuda()
  print("Loading VGG checkpoint '{}'".format(VGG_LOAD_EPOCH))
  checkpoint = torch.load(VGG_LOAD_EPOCH)
  start_epoch = checkpoint['epoch']
  best_prec1 = checkpoint['best_prec1']
  model_vgg16.load_state_dict(checkpoint['state_dict']) # Get frozen weights.

  # Set up PlacesCapsuleNet
  model_capsnet = PlacesCapsuleNet()
  print('Loading capsule net model from epoch:', CAPSNET_LOAD_EPOCH)
  state_dict = torch.load(CAPSNET_LOAD_EPOCH)
  model_capsnet.load_state_dict(state_dict)
  model_capsnet.cuda()

  print('Batch size:', BATCH_SIZE)
  print('Crop size:', CROP_SIZE)

  # Set up datasets.
  transforms_list = [transforms.CenterCrop(CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(DATA_MEAN, DATA_STD)]

  # Uncomment below for validating.
  # val_set = MiniPlacesDataset(os.path.abspath('./../../data/val.txt'),
  #                             os.path.abspath('./../../data/images/'),
  #                             transform=transforms.Compose(transforms_list))
  # val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  # validate(val_loader, [model_vgg16, model_capsnet], print_freq=10)

  # Uncomment for generating a submission file.
  generate_submission(model_vgg16, transforms_list)