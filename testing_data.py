from brian2 import *
from torch.utils.data import DataLoader

import numpy as np

import tonic
import tonic.transforms as transforms

import random

np.random.seed(211)
random.seed(211)

batch_size = 64
sensor_size = tonic.datasets.DVSGesture.sensor_size

#Define transformations
frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, n_time_bins = 20), transforms.DropEvent(p = 0.001)])

#Define training and test sets
DVS_train = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
DVS_test = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)

#Create dataloaders
trainloader = DataLoader(DVS_train, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)
testloader = DataLoader(DVS_test, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)

def generate_input_spikes(frames, dt):

    # find all nonzero entries (spikes)
    mask = frames > 0
    t_idx, c_idx, h_idx, w_idx = np.nonzero(mask)  # each is 1D array


    # map (c, h, w) → single neuron index 0..N_in-1
    neuron_indices = np.ravel_multi_index(
        (c_idx, h_idx, w_idx),
        dims=(2, 128, 128)
    )

    # map time bin index → Brian2 times
    spike_times = t_idx * dt

    return neuron_indices, spike_times

for frames, labels in trainloader:
    print(len(trainloader))
    print(len(frames))
