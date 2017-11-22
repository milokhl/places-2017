from __future__ import print_function, division
import os
import torch
from PIL import Image
from PIL.ImageOps import mirror, flip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MiniPlacesDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, get_flipped=False):
        """
        Args:
            txt_file (string): Path to the text file which associates image filenames with classes.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                self.lines.append(line)
        self.transform = transform
        self.root_dir = root_dir
        self.get_flipped = get_flipped
        assert(len(self.lines) > 0)

        print('Loaded MiniPlacesDataset.')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_line = self.lines[idx]
        img_path, label = img_line.split(' ')
        img_path = os.path.abspath(os.path.join(self.root_dir, img_path))
        
        label = label.replace('\n', '')
        label = int(label)

        image = Image.open(img_path)

        if self.get_flipped:
            image_horizontal_flip = mirror(image)
            image_vertical_flip = flip(image)

            if self.transform:
                image = self.transform(image)
                image_horizontal_flip = self.transform(image_horizontal_flip)
                image_vertical_flip = self.transform(image_vertical_flip)
                
            return image, image_horizontal_flip, image_vertical_flip, label

        else:
            if self.transform:
                image = self.transform(image)

            return image, label

class MiniPlacesTestSet(Dataset):
    """
    MiniPlaces dataset for the test set.
    The test set only has images, which are in the images_dir.
    There are no labels -- these are provided by the model.
    """
    def __init__(self, images_dir, transform=None, outfile='./predictions.txt', get_flipped=False):
        self.image_files = os.listdir(images_dir)
        self.image_files.sort()
        self.transform = transform
        self.images_dir = images_dir
        self.outfile = outfile
        self.get_flipped = get_flipped
        print('Loaded MiniPlaces test set from: %s' % self.images_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """ Returns a transformed image and filename. """
        image = Image.open(os.path.join(self.images_dir, self.image_files[idx]))

        # Return the image and it's flipped counterparts.
        if self.get_flipped:
            image_horizontal_flip = mirror(image)
            image_vertical_flip = flip(image)

            if self.transform:
                image = self.transform(image)
                image_horizontal_flip = self.transform(image_horizontal_flip)
                image_vertical_flip = self.transform(image_vertical_flip)
            return image, image_horizontal_flip, image_vertical_flip, self.image_files[idx]

        # Otherwise, just return the single image.
        else:
            if self.transform: image = self.transform(image)
            return image, self.image_files[idx]

    def write_labels(self, filename, labels):
        with open(self.outfile, 'a') as f:
            f.write('%s %d %d %d %d %d\n' % (filename, labels[0], labels[1], labels[2], labels[3], labels[4]))
