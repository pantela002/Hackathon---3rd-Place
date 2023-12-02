import numpy as np
from scipy import signal

from layer import Layer

class Convolutional(Layer):

    def __init__(self,input_shape,kernel_size,depth):
        # input_shape (black and white in my case, 28,28)
        # kernel size ( kernel size is kxk so kernel size is k)
        # depth (number of kernels)
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_size=(depth,input_height-kernel_size+1,input_width-kernel_size+1)
        self.kernels_size = (depth,input_depth,kernel_size,kernel_size)
        self.kernels=np.random.randn(*self.kernels_size)
        self.biases=np.random.randn(*self.output_size)


    def forward(self,input):
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j],self.kernels[i][j],"valid")

        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_size)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


