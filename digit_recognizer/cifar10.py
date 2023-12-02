import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from dense import Dense
from convolutional import Convolutional
import random
from reshape import Reshape
from activations import Softmax, Sigmoid
from losses import categorical_cross_entropy, categorical_cross_entropy_prime

"""def preprocess_data(x,y,limit):
    x = np.random.permutation(x)
    x=x.reshape(len(x),1,28,28)
    x=x.astype("float32") / 255.0
    y = to_categorical(y, num_classes=10)
    y=y.reshape(len(y),10,1)
    x=x[:limit]
    y=y[:limit]
    return x, y"""



def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    five_index = np.where(y == 3)[0][:limit]
    all_indices = np.hstack((zero_index, one_index,two_index,five_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 4, 1)
    return x, y

(x_train,y_train), (x_test, y_test) = mnist.load_data()
print(len(y_train))
print(len(x_train))

x_train, y_train = preprocess_data(x_train,y_train,10000)
print(len(y_train))
print(len(x_train))
x_test, y_test = preprocess_data(x_test, y_test,500)
network = [
    Convolutional((1,28,28),3,5),
    Sigmoid(),
    Reshape((5,26,26),(5*26*26,1)),
    Dense(5*26*26,100),
    Sigmoid(),
    Dense(100,4),
    Softmax()
]

epochs = 20
learning_rate =0.1


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output





errors = []

for e in range(epochs):
    error=0
    sf=len(x_train)
    sf1=0
    for x,y in zip(x_train,y_train):
        output=x
        for layer in network:
            output=layer.forward(output)
            

        error += categorical_cross_entropy(y,output)

        grad = categorical_cross_entropy_prime(y,output)

        for layer in reversed(network):
            grad=layer.backward(grad,learning_rate)

        if sf1%100==0:
            print("Training first "+str(sf1)+"/"+str(sf)+str(random.random())+str(random.random())+str(random.random())+str(random.random())+str(random.random()))
        sf1+=1

    error/= len(x_train)
    errors.append(error)
    print(f"{e+1}/{epochs}, error={error}")

#plt.plot(range(1, epochs + 1), errors, marker='o')
#plt.xlabel('Epochs')
#plt.ylabel('Error')
#plt.title('Training Error Over Epochs')
#plt.show()


# test
good=0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if np.argmax(output) == np.argmax(y):
        good+=1

print("Accuracy: ",good/len(x_test))

print(errors)






