import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import vgg_pytorch as VGG

from miniplaces_dataset import *

# Set up the data loader.
# TODO: add a dataset mean normalizer

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomSizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))

transform = transforms.Compose(
    [transforms.Resize((224, 224), interpolation=2),
     transforms.ToTensor(),
     transforms.Normalize((0.45834960097,0.44674252445,0.41352266842), (0.5, 0.5, 0.5))])

training_set = MiniPlacesDataset('/home/milo/envs/tensorflow35/miniplaces/data/train.txt',
                                 '/home/milo/envs/tensorflow35/miniplaces/data/images/',
                                 transform=transform)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=True, num_workers=2)

# Define the model, loss, and optimizer.
vgg11 = VGG.vgg11(num_classes=100)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg11.parameters(), lr=0.001, momentum=0.9)

# Parameters
epochs = 10

for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        print(type(inputs), type(labels))
        print(inputs.size())

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg11(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
