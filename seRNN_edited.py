import torch, torch.nn as nn
from torch.utils.data import DataLoader
import plotly.graph_objects as go


import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

import numpy as np

import scipy
from scipy.stats import pearsonr

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset

import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
import pandas as pd

import networkx as netx
from networkx.algorithms.community import greedy_modularity_communities, modularity
import csv

np.random.seed(211)
random.seed(211)
torch.manual_seed(211)

batch_size = 64
dtype = torch.float
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sensor_size = tonic.datasets.DVSGesture.sensor_size

#Define transformations
frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, n_time_bins = 20), transforms.DropEvent(p = 0.001)])

#Define training and test sets
DVS_train = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
DVS_test = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)

#Create dataloaders
trainloader = DataLoader(DVS_train, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)
testloader = DataLoader(DVS_test, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)
  

#Membrane parameters
tau_mem = 20e-3
dist_shape = 3
time_step = 0.5e-3

#Clipping function
def clip_tc(x):
    clipped_tc = x.clamp_(0.7165, 0.995)
    return clipped_tc

#Initialize membrane time constant distribution
def init_tc():
    dist_gamma = np.random.gamma(dist_shape, tau_mem / 3, 100)
    dist_beta = torch.from_numpy(np.exp(-time_step / dist_gamma))
    clipped_beta = clip_tc(dist_beta)
    return clipped_beta


#Size parameters
num_inputs = 128*128*2
num_hidden = 100
num_outputs = 11

#Network parameters
het_tau = init_tc().to(device)
hom_tau = 0.9753

#Optimization mechanism
spike_grad = surrogate.fast_sigmoid(slope = 100)

#Model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.RLeaky(beta = het_tau, linear_features = num_hidden, learn_beta = True, spike_grad = spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta = hom_tau, spike_grad = spike_grad)

    def forward(self, x):

        #Initialize parameters
        spk1, mem1 = self.lif1.init_rleaky()
        mem2 = self.lif2.init_leaky()

        #Record output layer
        spk_out_rec = []
        mem_out_rec = []

        #Forward loop
        for step in range(x.size(0)):
            batched_data = x[step].view(batch_size, num_inputs)
            cur1 = self.fc1(batched_data)
            spk1, mem1 = self.lif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out_rec.append(spk2)
            mem_out_rec.append(mem2)

        #Convert output lists to tensors
        spk_out_rec = torch.stack(spk_out_rec)
        mem_out_rec = torch.stack(mem_out_rec)

        return spk_out_rec, mem_out_rec

net = Net().to(device)

tc_hist = []
pretrain_tau = (-time_step / np.log(het_tau.cpu())) / 1e-3
tc_hist.append(pretrain_tau.numpy())

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, betas = (0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate = 0.8, incorrect_rate = 0.2)

#Distance matrix -----
network_structure = [10, 2, 10]
distance_metric = 'euclidean'
distance_power = 1

#Define each dimension's neuron
ny = np.array([1])
nx = np.arange(network_structure[0])
nz = np.arange(network_structure[2])

#Create coordinate grid
[x,y,z] = np.meshgrid(nx,ny,nz)
coordinates = [x.ravel(),y.ravel(),z.ravel()]

#Calculate Euclidean distance matrix
euclidean_vector = scipy.spatial.distance.pdist(np.transpose(coordinates), metric=distance_metric)
euclidean = scipy.spatial.distance.squareform(euclidean_vector**distance_power)
distance_matrix = euclidean.astype('float64')

distance_matrix = torch.from_numpy(distance_matrix).to(device)

inputs_square = [10, 0, 10] #this is the square that the inputs will be mapped to. is the same size as hidden layer
nxi = np.linspace(0, 9, 128)
nyi = np.array([0])
nzi = np.linspace(0, 9, 128)

[xi,yi,zi] = np.meshgrid(nxi, nyi, nzi)
input_coordinates = np.stack([xi.ravel(),yi.ravel(),zi.ravel()], axis=1)
input_coords = np.concatenate([input_coordinates, input_coordinates], axis=0)

#calculate euclidian distance matrix between inputs and hidden layer
in_euc_vect = scipy.spatial.distance.cdist(input_coords, np.stack(coordinates, axis=1), metric=distance_metric) #dimensions 32768 X 100
input_distance_matrix = in_euc_vect.astype('float64')



#Test for spatial regularization
def test_euclidean(x, y):
    x = torch.abs(x)
    x_array = x.detach().cpu().numpy()
    flat_x_array = x_array.flatten()
    y = torch.abs(y)
    y_array = y.detach().cpu().numpy()
    flat_y_array = y_array.flatten()
    correlation = pearsonr(flat_x_array, flat_y_array)[0]
    return correlation

print(f"Initial, pre-training correlation between distance and weight matrices (should be approx. 0): {test_euclidean(distance_matrix, net.lif1.recurrent.weight)}")
     
#Training parameters
num_epochs = 25
comms_factor = 1

#Regularization parameters
regu_strength = 0.5e-1

sig = 0 #effective connecting distance of 6
#Training loop
for run in range(15):
    csv_file = open(f"training_run_{run}.csv", mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "regularization_term", "modularity_Q"])

    train_loss_hist = []
    train_acc_hist = []
    rec_tot_hist = []
    corr_hist = []
    test_acc_hist = []
    test_loss_hist = []
    weight_matrix = []

    rec_tot_hist.append(torch.sum(torch.abs(net.lif1.recurrent.weight.detach())))
    corr_hist.append(test_euclidean(distance_matrix, net.lif1.recurrent.weight))

    if sig > 0:
        gaus_weights = np.exp(- (input_distance_matrix**2) / (2 * sig**2))

        #scaling weights
        a = 1/ np.sqrt(num_inputs)
        scale = a/gaus_weights.max() #0.8 is max value returned by gaus_weights
        rands = np.random.uniform(-1.0, 1.0, size=gaus_weights.shape) * scale
        fc1_weights = gaus_weights * rands #dimensions: input X hidden

        #assign to net
        with torch.no_grad():
            net.fc1.weight.copy_(torch.tensor(fc1_weights.T, dtype=net.fc1.weight.dtype))
            net.fc1.bias.zero_()

    for epoch in range(1, num_epochs + 1):
        for i, (data, targets) in enumerate(iter(trainloader)):
            # Load data on CUDA
            data = data.to(device)
            targets = targets.to(device)

            # Set model to training mode
            net.train()
            spk_outputs, mem_outputs = net(data)

            # Create absolute weight matrix (don't detach)
            abs_weight_matrix = torch.abs(net.lif1.recurrent.weight)

            # Calculate communicability (detach to prevent complex gradient issues)
            with torch.no_grad():
                step1 = torch.sum(abs_weight_matrix, dim=1)
                step2 = torch.pow(step1, -0.5)
                step3 = torch.diag(step2)
                step4 = torch.linalg.matrix_exp(step3 @ abs_weight_matrix @ step3)
                comms_matrix = step4.fill_diagonal_(0)
                comms_matrix = comms_matrix ** comms_factor

            # Calculate regularization term
            regularization_term = torch.sum(abs_weight_matrix * distance_matrix * comms_matrix.detach())

            # Calculate total loss
            task_loss = loss_fn(spk_outputs, targets)
            loss_val = task_loss + regu_strength * regularization_term

            # Gradient calculation and weight updates
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            clip_tc(net.lif1.beta.detach())

            # Store loss history
            train_loss_hist.append(loss_val.item())

        #Evaluations (every epoch)
        net.eval()

        #Training accuracy
        acc = SF.accuracy_rate(spk_outputs, targets)
        train_acc_hist.append(acc)

        #Sum of regularized weights
        rec_tot = torch.sum(torch.abs(net.lif1.recurrent.weight.detach()))
        rec_tot_hist.append(rec_tot)

        #Correlation of distance and weight matrices
        corr_matrix = test_euclidean(distance_matrix, net.lif1.recurrent.weight.detach())
        corr_hist.append(corr_matrix)

        #Save membrane time constant matrix
        converted_tc = (-time_step / np.log(net.lif1.beta.cpu().detach())) / 1e-3
        tc_hist.append(converted_tc.numpy())

        #Save weight matrix
        weight_matrix.append(net.lif1.recurrent.weight.detach().cpu())

        #Validation accuracy
        with torch.no_grad():
            net.eval()
            total = 0
            correct = 0

            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device)

                test_spk, test_mem = net(data)

                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                test_loss = loss_fn(test_spk, targets) + regu_strength * regularization_term

            test_acc_hist.append(correct / total)
            test_loss_hist.append(test_loss.item())

            G = netx.from_numpy_array(net.lif1.recurrent.weight.detach().cpu().numpy())

            # Detect communities (e.g. Louvain or greedy modularity)
            communities = list(greedy_modularity_communities(G))

            # Compute Newman's modularity Q
            Q = modularity(G, communities, weight='weight')

        #Print statements
        #if epoch % 5 == 0:
        print(f"Epoch {epoch}/{num_epochs} === Train loss: {loss_val.item():.2f} --- ", end = "")
        print(f"Train accuracy: {acc * 100:.2f}% --- ", end = "")
        print(f"Val. loss: {test_loss.item():.2f} --- ", end = "")
        print(f"Val. accuracy: {100 * correct / total:.2f}% --- ", end = "")
        print(f"Regularization term: {regularization_term.item():.4f}", end = "")
        print(f"Modularity: {Q:.4f}")

        writer.writerow([
        epoch,
        loss_val.item(),
        acc,
        test_loss.item(),
        correct / total,
        regularization_term.item(),
        Q
        ])
        csv_file.flush()

    csv_file.close()
    sig += 1

