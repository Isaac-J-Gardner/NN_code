from brian2 import *
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np

import tonic
import tonic.transforms as transforms

import random

np.random.seed(211)
random.seed(211)

print(1)

batch_size = 64
n_bins = 20
bin_dt = 10*ms
T_trial = n_bins * bin_dt
sensor_size = tonic.datasets.DVSGesture.sensor_size

#Define transformations
frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, n_time_bins = n_bins), transforms.DropEvent(p = 0.001)])

#Define training and test sets
DVS_train = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
DVS_test = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)

#Create dataloaders
trainloader = DataLoader(DVS_train, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)
testloader = DataLoader(DVS_test, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)
