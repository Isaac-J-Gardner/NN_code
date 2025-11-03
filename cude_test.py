import numpy as np
import scipy

network_structure = [10, 2, 10]
distance_metric = 'euclidean'
distance_power = 1

#Define each dimension's neurons
nx = np.linspace(0, 9, 10)
ny = np.array([2])
nz = np.linspace(0, 9, 10)

#Create coordinate grid
[x,y,z] = np.meshgrid(nx,ny,nz) #coordinates now go from 0-9, neurons sitting on each integer
hidden_coordinates = np.stack([x.ravel(),y.ravel(),z.ravel()], axis=1)

print(hidden_coordinates.shape)

#Calculate Euclidean distance matrix
euclidean_vector = scipy.spatial.distance.pdist(np.transpose(hidden_coordinates), metric=distance_metric)
euclidean = scipy.spatial.distance.squareform(euclidean_vector**distance_power)
distance_matrix = euclidean.astype('float64')

inputs_square = [10, 0, 10] #this is the square that the inputs will be mapped to. is the same size as hidden layer
nxi = np.linspace(0, 9, 128)
nyi = np.array([0])
nzi = np.linspace(0, 9, 128)

[xi,yi,zi] = np.meshgrid(nxi, nyi, nzi)
input_coordinates = np.stack([xi.ravel(),yi.ravel(),zi.ravel()], axis=1)
input_coords = np.concatenate([input_coordinates, input_coordinates], axis=0)

print(input_coords.shape)

#calculate euclidian distance matrix between inputs and hidden layer
in_euc_vect = scipy.spatial.distance.cdist(input_coords, hidden_coordinates, metric=distance_metric)
input_distance_matrix = in_euc_vect.astype('float64')
sig = 3.0 #effective connecting distance of 6
gaus_weights = np.exp(- (input_distance_matrix**2) / (2 * sig**2))
print(gaus_weights.min())
print(gaus_weights.max())
a = 1/ np.sqrt(32768)
scale = a/gaus_weights.max() #0.8 is max value returned by gaus_weights
rands = np.random.uniform(-1.0, 1.0, size=gaus_weights.shape) * scale
fc1_weights = gaus_weights * rands
print(fc1_weights.min())
print(fc1_weights.max())


