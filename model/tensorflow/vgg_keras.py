from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import metrics

from DataLoader import *
import inspect
import os

import tensorflow as tf
import time

# Dataset Parameters
batch_size = 1024
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
num_classes = 100

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 1
step_save = 10000
path_save = 'vgg16'
start_from = ''

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
}
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
}

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

# weights: 'None' to train on randomly initialized weights, or 'imagenet'
# include_top: whether to include the top 3 fully connected layers of the model
# pooling: 'avg' uses GlobalAveragePooling2D, which returns a 2D tensor
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

optimizer = Adam(lr=learning_rate)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['top_k_categorical_accuracy', 'categorical_accuracy'])

def convert_labels_categorical(labels_batch):
    labels = np.zeros((batch_size, num_classes))
    for i in range(len(labels_batch)):
        labels[i][int(labels_batch[i])] = 1
    return labels

# At each iteration, load a batch into memory
# And train for some number of epochs on it
gpu_size=12 # largest batch we can fit on GPU
for t_iter in range(training_iters):
    images_batch, labels_batch = loader_train.next_batch(batch_size)
    labels = convert_labels_categorical(labels_batch) # need to make this into a one-hot vector
    model.fit(x=images_batch, y=labels, batch_size=gpu_size, epochs=1, verbose=1)

    # At the end of this training iteration, test on the validation set.
    images_val, labels_val = loader_val.next_batch(batch_size)
    labels_val_categorical = convert_labels_categorical(labels_val)
    model.evaluate(x=images_val, y=labels_val_categorical, batch_size=gpu_size, verbose=1)