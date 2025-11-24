import numpy as np
import scipy

inputs_square = [10, 0, 10] #this is the square that the inputs will be mapped to. is the same size as hidden layer
nxi = np.linspace(0, 9, 128)
nyi = np.array([0])
nzi = np.linspace(0, 9, 128)

[xi,yi,zi] = np.meshgrid(nxi, nyi, nzi)
input_coordinates = np.stack([xi.ravel(),yi.ravel(),zi.ravel()], axis=1)
input_coords = np.concatenate([input_coordinates, input_coordinates], axis=0)

print(input_coords)
print(input_coords[:, 2])
x = input_coords[:, 0]
print(x)
