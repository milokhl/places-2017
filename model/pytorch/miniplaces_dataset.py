from __future__ import print_function, division
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MiniPlacesDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
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

        if self.transform:
            image = self.transform(image)

        return image, label