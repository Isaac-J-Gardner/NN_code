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

print(2)

def generate_input_spikes(input_frames, dt):
    frames = input_frames.detach().cpu().numpy()  # shape [T, C, H, W]

    # find all nonzero entries (spikes)
    mask = frames > 0
    t_idx, c_idx, h_idx, w_idx = np.nonzero(mask)  # obtaining index of pixels>0

    # map (c, h, w) → single neuron index
    input_neuron_indices = np.ravel_multi_index(
        (c_idx, h_idx, w_idx),
        dims=(C, H, W)
    ) #indexes below 16384 are "off" (C = 0, decreasing brightness) neurons, indexes above are "on" (increasing brightness) neurons

    # map time bin index → Brian2 times
    spike_times = t_idx * dt

    return input_neuron_indices, spike_times

def generate_teach_spikes(label, n_spikes):

    teach_idx = np.full(n_spikes, int(label), dtype=int)

    teach_times = T_trial - np.arange(n_spikes, 0, -1) * bin_dt #teacher spikes will occur at the end of each gesture for a selected number of frames n_spikes

    return teach_idx, teach_times

C, H, W = 2, 128, 128
N_in = C * H * W

N_hid = 100
N_out = 11

taupre = taupost = 20*ms
V_th = 1*volt
V_r = 0*volt
wmax = 0.001
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05
conn_decay = 6*metre

tau = 10*ms
eqs = '''
dv/dt = -v/tau : volt 
x : metre
y : metre
z : metre'''

print(3)

input_layer = NeuronGroup(N_in, eqs, threshold='v>V_th', reset='v = V_r', method='exact')
hidden_layer = NeuronGroup(N_hid, eqs, threshold='v>V_th', reset='v = V_r', method='exact')
output_layer = NeuronGroup(N_out, eqs, threshold='v>V_th', reset='v = V_r', method='exact')

nxi = np.linspace(0, 9, W)
nyi = np.linspace(0, 9, H)
nzi = np.array([0])

[xi,yi,zi] = np.meshgrid(nxi, nyi, nzi)
input_coords = np.stack([xi.ravel(),yi.ravel(),zi.ravel()], axis=1)
input_coords = np.concatenate([input_coords, input_coords], axis=0)

#Define each dimension's neuron
nxh = np.linspace(0, 9, 10)
nyh = np.linspace(0, 9, 10)
nzh = np.array([1])

#Create coordinate grid
[xh,yh,zh] = np.meshgrid(nxh,nyh,nzh)
hidden_coords = np.stack([xh.ravel(),yh.ravel(),zh.ravel()], axis=1)

hidden_layer.x = hidden_coords[:, 0]*metre
hidden_layer.y = hidden_coords[:, 1]*metre
hidden_layer.z = hidden_coords[:, 2]*metre

print(4)

input_layer.x = input_coords[:, 0]*metre
input_layer.y = input_coords[:, 1]*metre
input_layer.z = input_coords[:, 2]*metre

in_hid = Synapses(input_layer, hidden_layer, 
                '''
            w : 1
            dapre/dt = -apre/taupre : 1 (event-driven)
            dapost/dt = -apost/taupost : 1 (event-driven)
            ''',
            on_pre='''
            v_post += w * volt
            apre += Apre
            w = clip(w+apost, 0, wmax)
            ''',
            on_post='''
            apost += Apost
            w = clip(w+apre, 0, wmax)'''
                )

hid_hid = Synapses(hidden_layer, hidden_layer, 
                     '''
                    w : 1
                    dapre/dt = -apre/taupre : 1 (event-driven)
                    dapost/dt = -apost/taupost : 1 (event-driven)
                    ''',
                    on_pre='''
                    v_post += w * volt
                    apre += Apre
                    w = clip(w+apost, 0, wmax)
                    ''',
                    on_post='''
                    apost += Apost
                    w = clip(w+apre, 0, wmax)'''
                     )

hid_out = Synapses(hidden_layer, output_layer, 
                     '''
                    w : 1
                    dapre/dt = -apre/taupre : 1 (event-driven)
                    dapost/dt = -apost/taupost : 1 (event-driven)
                    ''',
                    on_pre='''
                    v_post += w * volt
                    apre += Apre
                    w = clip(w+apost, 0, wmax)
                    ''',
                    on_post='''
                    apost += Apost
                    w = clip(w+apre, 0, wmax)'''
                     )

in_hid.connect(p = 'exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/(2*conn_decay**2))')

hid_hid.connect(p = 'exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/(2*conn_decay**2))')

hid_out.connect(p = 1)

in_hid.w  = '0.0005*rand()'   # initial weights around 0–0.05
hid_hid.w = '0.001*rand()'
hid_out.w = '0.02*rand()'

print(5)

temp=0
for frames, labels in trainloader:
    temp += 1
    print("frame =", temp)
    T, B, C, H, W = frames.shape
    for b in range(B):
        print(6)
        input_frames = frames[:, b]
        output = labels[b].item()

        input_index, input_times = generate_input_spikes(input_frames, bin_dt)
        teach_index, teach_times = generate_teach_spikes(output, n_bins) #all frames receive training

        inputs = SpikeGeneratorGroup(N_in, input_index, input_times)

        S_inputs = Synapses(inputs, input_layer, on_pre='v_post += 2.0 * volt')
        S_inputs.connect(j='i')

        print(7)

        teacher_layer = SpikeGeneratorGroup(N_out, teach_index, teach_times)

        S_teach = Synapses(teacher_layer, output_layer, on_pre='v_post += 2.0 * volt') #strong enough to cause a spike
        S_teach.connect(j='i') #1 to 1 mapping

        M1 = SpikeMonitor(input_layer)
        M2 = SpikeMonitor(hidden_layer)
        M3 = SpikeMonitor(output_layer)

        print(8)

        run(250*ms)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

        axes[0].plot(M1.t/ms, M1.i, '.', ms=1)
        axes[0].set_ylabel('Neuron index')
        axes[0].set_title('Input Layer')

        axes[1].plot(M2.t/ms, M2.i, '.', ms=1)
        axes[1].set_ylabel('Neuron index')
        axes[1].set_title('Hidden Layer')

        axes[2].plot(M3.t/ms, M3.i, '.', ms=1)
        axes[2].set_ylabel('Neuron index')
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_title('Output Layer')

        plt.tight_layout()
        plt.show()
