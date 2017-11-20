# Submission format: <filename> <label(1)> <label(2)> <label(3)> <label(4)> <label(5)>
# Example line: val/00000005.jpg 65 3 84 93 67

# Your team, LaCroixNet, has been registered for the challenge.
# Your team code (case-sensitive) for submitting to the leaderboard is: fFR7jWuG2XqiAImnJQrN
from __future__ import print_function, division
import os, sys, time

sys.path.append('../')

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import vgg_pytorch as VGG
from miniplaces_capsnet import PlacesCapsuleNet

from miniplaces_dataset import *
from utils import accuracy, AverageMeter, save_checkpoint, log

def main():
	BATCH_SIZE = 64
	VGG_LOAD_EPOCH = './convnet/model_best.pth.tar'
	CAPSNET_LOAD_EPOCH = './capsnet/epochs/epoch_16.pt'

	# Apply a series of transformations to the input data.
	DATA_MEAN = (0.45834960097, 0.44674252445, 0.41352266842)
	DATA_STD = (0.229, 0.224, 0.225)
	CROP_SIZE = 120
	batch_size = 200

	# Set up VGG16
	model_vgg16 = VGG.vgg16(num_classes=100, dropout=0.5, light=True)
	model_vgg16.features = torch.nn.DataParallel(model_vgg16.features)
	model_vgg16.cuda()
	model_vgg16.eval()

	print("Loading VGG checkpoint '{}'".format(VGG_LOAD_EPOCH))
	checkpoint = torch.load(VGG_LOAD_EPOCH)
	start_epoch = checkpoint['epoch']
	best_prec1 = checkpoint['best_prec1']
	model_vgg16.load_state_dict(checkpoint['state_dict']) # Get frozen weights.

	# Set up the PlacesCapsuleNet
	model_capsnet = PlacesCapsuleNet()
	print('Loading capsule net model from epoch:', LOAD_EPOCH)
	model_capsnet.load_state_dict(torch.load(LOAD_EPOCH))
	model_capsnet.cuda()
	model_capsnet.eval()

	print('Batch size:', batch_size)
	print('Crop size:', CROP_SIZE)

	# Load in the validation set.
	val_set = MiniPlacesDataset(os.path.abspath('./../../data/val.txt'),
	                            os.path.abspath('./../../data/images/'),
	                            transform=transforms.Compose([
	                            transforms.CenterCrop(CROP_SIZE),
	                            transforms.ToTensor(),
	                            transforms.Normalize(DATA_MEAN, DATA_STD)]))

	val_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

	validate(val_loader, [model_vgg16, model_capsnet], print_freq=10)


def validate(val_loader, models, print_freq=1):
  print('Starting validation of ensemble!')
  model1_weight = 0.5
  model2_weight = 0.5

  batch_time = AverageMeter()
  top1_ensemble = AverageMeter()
  top5_ensemble = AverageMeter()

  top1_model1 = AverageMeter()
  top1_model2 = AverageMeter()

  top5_model1 = AverageMeter()
  top5_model2 = AverageMeter()

  end = time.time()
  for i, (input, target) in enumerate(val_loader):
      target = target.cuda(async=True)
      input_var = torch.autograd.Variable(input, volatile=True)
      target_var = torch.autograd.Variable(target, volatile=True)

      # Get outputs from each model in ensemble.
      outputs = [model(input_var) for model in models]

      # Weight each model and combine outputs.
      ensemble_output = model1_weight * outputs[0] + model2_weight * outputs[1]

      # Measure accuracies.
      prec1 = [accuracy(output.data, target, topk=(1)) for output in outputs]
      prec5 = [accuracy(output.data, target, topk=(5)) for output in outputs]
      top1_model1.update(prec1[0][0], input.size(0))
      top1_model2.update(prec1[1][0], input.size(0))
      top5_model1.update(prec5[0][0], input.size(0))
      top5_model2.update(prec5[1][0], input.size(0))

      # Get the enemble accuracies.
      prec1_ensemble, prec5_ensemble = accuracy(ensemble_output.data, target, topk=(1,5))
      top1_ensemble.update(prec1_ensemble[0], input.size[0])
      top5_ensemble.update(prec5_ensemble[0], input_size[0])

      # Measure time.
      batch_time.update(time.time() - end)
      end = time.time()

      # Print statistacks.
      if i % print_freq == 0:
          print_str = 'Validation: [{0}/{1}]\t'\
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
            'Prec@1 (ens) {top1_ensemble.val:.3f} ({top1_ensemble.avg:.3f})\t'\
            'Prec@5 (ens) {top5_ensemble.val:.3f} ({top5_ensemble.avg:.3f})\t \
            Prec@1 (1) {top1_model1.val:.3f} ({top1_model1.avg:.3f})\t \
            Prec@5 (1) {top5_model1.val:.3f} ({top5_model1.avg:.3f})\t \
            Prec@1 (2) {top1_model2.val:.3f} ({top1_model2.avg:.3f})\t \
            Prec@5 (2) {top5_model2.val:.3f} ({top5_model2.avg:.3f})\t'.format(
             i, len(val_loader), batch_time=batch_time,
             top1_ensemble=top1_ensemble, top5_ensemble=top5_ensemble,
             top1_model1=top1_model1, top5_model1=top5_model1,
             top1_model2=top1_model2, top5_model2=top5_model2)
          print(print_str)

  print_str = ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_ensemble, top5=top5_ensemble)
  print(print_str)

  print('Finished validation!')
  return top1_ensemble.avg, top5_ensemble.avg

if __name__ == '__main__':
	main()