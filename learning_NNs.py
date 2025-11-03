import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

class Loss:
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class cat_loss(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)

        neg_log = -np.log(correct_confidences)
        return neg_log
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        labels = len(dvalues[0]) #this generalises it for different types of data, we have 3 labels for spiral

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] #genius this is, eye(x) returns an x*x identity matrix, y_true (for spiral) is either 0, 1, or 2, and thus this returns an
                                            #array constisting of [1,0,0], [0,1,0], or [0,0,1] depending on the label y_true was 
        
        self.dinputs = -y_true/dvalues

        self.dinputs = self.dinputs/samples

class activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] = 0

class activation_softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probs = exp_values/np.sum(exp_values, axis=1, keepdims=True)

        self.output = probs

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacob = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacob, single_dvalues)

class act_soft_loss_cat:

    def __init__(self):
        self.activation = activation_softmax()
        self.loss = cat_loss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)

        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1 #y_true is 1 in the index of the correct classification. converting y_true to a list, we can simply subtract 1 from the correct prediction
        self.dinputs = self.dinputs/samples

class layer_dense:

    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons) #outputs an array of dimensions n_inputsXn_neurons (so that we don't have to transpose later)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class SGD_optimiser:

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

X, y = spiral_data(samples = 100, classes=3)

dense1 = layer_dense(2,64)

activation1 = activation_ReLU()

dense2 = layer_dense(64,3)

loss_activation = act_soft_loss_cat()

opt = SGD_optimiser(0.2)

for epoch in range(100001):
 
    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch%1000:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')

    loss_activation.backward(loss_activation.output, y)

    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)

    dense1.backward(activation1.dinputs)

    opt.update_params(dense1)
    opt.update_params(dense2)