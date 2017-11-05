# Submission format: <filename> <label(1)> <label(2)> <label(3)> <label(4)> <label(5)>
# Example line: val/00000005.jpg 65 3 84 93 67

# Your team, LaCroixNet, has been registered for the challenge.
# Your team code (case-sensitive) for submitting to the leaderboard is: fFR7jWuG2XqiAImnJQrN

from __future__ import print_function, division
import os, sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class MiniPlacesTestSet(Dataset):
  """
  MiniPlaces dataset for the test set.
  The test set only has images, which are in the images_dir.
  There are no labels -- these are provided by the model.
  """
  def __init__(self, images_dir, transform=None):
    self.image_files = os.listdir(images_dir)
    self.transform = transform
    self.images_dir = images_dir
    print('Loaded MiniPlaces test set from: %s' % self.images_dir)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    image = Image.open(self.image_files[idx])
    if self.transform: image = self.transform(image)
    return image

  def write_labels(self, indexes, labels):
  	pass

# Apply same transforms to the test set as the training set, except without randomized augmentation.
transform = transforms.Compose(
    [transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.45834960097, 0.44674252445, 0.41352266842), (0.229, 0.224, 0.225))])

batch_size = 58 # Run out of memory at 64...
test_set = MiniPlacesDataset('/home/milo/envs/tensorflow35/miniplaces/data/train.txt',
                                 '/home/milo/envs/tensorflow35/miniplaces/data/images/',
                                 transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=10)