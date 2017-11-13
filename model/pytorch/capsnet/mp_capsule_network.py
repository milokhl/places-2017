"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""

# TODO
# figure out how to get data into engine
# figure out what params need to be change to accomodate 100 classes
# figure out if reconstruction will work / be helpful
# add more caps layers
# add more augmentation

import sys, os

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

sys.path.append('../')
from miniplaces_dataset import *

BATCH_SIZE = 16 # TODO
NUM_CLASSES = 100
NUM_EPOCHS = 500 # TODO
NUM_ROUTING_ITERATIONS = 3 # TODO
CROP_SIZE = 64

DATA_MEAN = (0.45834960097, 0.44674252445, 0.41352266842)
DATA_STD = (0.229, 0.224, 0.225)


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    """
    Randomly shifts the input image by up to 2 pixels in any direction.
    """
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class PlacesCapsuleNet(nn.Module):
    def __init__(self):
        """
        Size of convolution output given by: (W−F+2P) / S + 1.
        # W: input volume size
        # F: receptive field of conv neurons
        # S: stride
        # P: padding
        """
        # Conv1 Params
        conv1_filters = 256
        conv1_kernel_size = 9
        conv1_stride = 2
        conv1_size = (CROP_SIZE - conv1_kernel_size) // conv1_stride + 1

        # Primary Capsule Params
        cap1_units = 8
        cap1_out_channels = 32
        cap1_kernel_size = 9
        cap1_stride = 2
        conv2_size = (conv1_size - cap1_kernel_size) // cap1_stride + 1

        # Secondary Capsule Params
        cap2_units = 12
        cap2_out_channels = 16

        # Category Capsule Params
        category_out_channels = 32

        super(PlacesCapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_filters, kernel_size=conv1_kernel_size,
                               stride=conv1_stride)

        self.primary_capsules = CapsuleLayer(num_capsules=cap1_units, num_route_nodes=-1, in_channels=conv1_filters,
                                             out_channels=cap1_out_channels, kernel_size=cap1_kernel_size, stride=cap1_stride)

        self.digit_capsules = CapsuleLayer(num_capsules=cap2_units, num_route_nodes=cap1_out_channels * conv2_size * conv2_size,
                                           in_channels=cap1_units, out_channels=cap2_out_channels)

        self.category_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=cap2_units,
                                              in_channels=cap2_out_channels, out_channels=category_out_channels)

        print('Initialized PlacesCapsuleNet!')

    def forward(self, x, y=None):
        # Note: the transpose is needed to flip the batch size into the 0th dimension.
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        x = self.category_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = None # Disabled for now.

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss / images.size(0) # Removed reconstruction loss for now.


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    from tqdm import tqdm
    import torchnet as tnt

    # Create the model.
    model = PlacesCapsuleNet()

    # Load from checkpoint here is needed.
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    print("Model Parameters:", sum(param.numel() for param in model.parameters()))
    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    # Create a bunch of loggers that can be viewed in the browser.
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()


    def get_iterator(mode):
        """
        Returns an iterable TensorDataset.
        @param mode (bool) True for training mode, False for testing mode.
        """
        # dataset = MNIST(root='./data', download=True, train=mode) # TODO
        training_transforms = transforms.Compose(
            [transforms.Resize(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEAN, DATA_STD)]
        )

        validation_transforms = transforms.Compose([
            transforms.Resize(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEAN, DATA_STD)])

        if mode:
            dataset = MiniPlacesDataset(os.path.abspath('./../../../data/train.txt'),
                                        os.path.abspath('./../../../data/images/'),
                                        transform=training_transforms)
        else:
            dataset = MiniPlacesDataset(os.path.abspath('./../../../data/val.txt'),
                                        os.path.abspath('./../../../data/images/'),
                                        transform=validation_transforms)

        # Replaced the TensorDataset with our own custom dataset.
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=mode, num_workers=4)

    # This function takes in a sample of data and outputs the loss and outputs of the network.
    # The engine calls this function during training and testing loops.
    def processor(sample):
        data, labels, training = sample
        labels = torch.LongTensor(labels)
        labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        # Testing does not involve groundtruth labels.
        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()

    # Called every time the training loop requests a new batch.
    def on_sample(state):
        state['sample'].append(state['train'])

    # Called every time a batch is feed forward through the model.
    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    # Called at the start of each new epoch.
    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    # Called at the end of every epoch.
    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        # Validate the model after every training epoch.
        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start

    # Set up hooks for the engine.
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    # network, iterator, maxepoch, optimizer
    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
