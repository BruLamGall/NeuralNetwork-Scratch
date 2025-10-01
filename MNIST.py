import numpy as np
import pandas as pd

'''Load and preprocess data'''
print('Loading data...')
data = pd.read_csv('train.csv') # This file can't be uploaded to gitgub, you can find it on kaggle
data = np.array(data)         
m, n = data.shape

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255.



def init_params():
    '''Creates parameters for the neural network'''
    w1 = np.random.rand(16, 784) - 0.5
    b1 = np.random.rand(16, 1)   - 0.5
    w2 = np.random.rand(16, 16)  - 0.5
    b2 = np.random.rand(16, 1)   - 0.5
    w3 = np.random.rand(10, 16)  - 0.5
    b3 = np.random.rand(10, 1)   - 0.5
    return w1, b1, w2, b2, w3, b3

def ReLU(z):
    '''Applies Rectified Linear Unit activation function'''
    return np.maximum(z, 0)

def softmax(z):
    '''Applies the Softmax activation function'''
    a = np.exp(z) / sum(np.exp(z))
    return a
    
def ReLU_deriv(z):
    '''Calculates the derivative of the ReLU function'''
    return z > 0

def one_hot(y):
    '''Converts a vector to a one-hot encoded matrix'''
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def forward_prop(w1, b1, w2, b2, w3, b3, x):
    '''
    Performs the forward propagation of the neural network
    
    Args:
        w1 (np.array): Weight matrix for the input to hidden 1 step
        b1 (np.array): Bias vector for the hidden 1 layer
        w2 (np.array): Weight matrix for the hidden 1 to hidden 2 step
        b2 (np.array): Bias vector for the hidden 2 layer
        w3 (np.array): Weight matrix for the hidden 2 to output step
        b3 (np.array): Bias vector for the output layer
        x  (np.array): Input data
    
    Returns:
        tuple: z1, a1, z2, a2, z3, a3 (intermediate and output values with and without activation function)
    '''
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def backward_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, x, y):
    '''
    Performs the backward propagation of the neural network.

    Args:
        y  (np.array): Expected output data.
    
    Returns:
        tuple: dw1, db1, dw2, db2, dw3, db3 (gradients for all the neural network's parameters).
    '''
    one_hot_y = one_hot(y)
    dz3 = a3 - one_hot_y
    dw3 = 1 / m * dz3.dot(a2.T)
    db3 = 1 / m * np.sum(dz3, axis=1, keepdims=True)
    dz2 = w3.T.dot(dz3) * ReLU_deriv(z2)
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2, dw3, db3

def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, lr):
    '''
    Updates the params following the gradient descent formula <New -= Learning rate * Gradient>

    Args:
        lr (float): The learning rate
    
    Returns:
        tuple: Updated w1, b1, w2, b2, w3, b3
    '''
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1    
    w2 = w2 - lr * dw2  
    b2 = b2 - lr * db2    
    w3 = w3 - lr * dw3
    b3 = b3 - lr * db3
    return w1, b1, w2, b2, w3, b3

def get_accuracy(a3, y):
    '''Calculates the accuracy of the model'''
    predictions = np.argmax(a3, 0)
    return np.sum(predictions == y) / y.size 

def gradient_descent(x, y, x_dev, y_dev, lr, iterations):
    '''
    Performs gradient descent to train the neural network

    Returns:
        tuple: trained parameters w1, b1, w2, b2, w3, b3
    '''
    w1, b1, w2, b2, w3, b3 = init_params()
    print('Starting Training...')
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, x)
        dw1, db1, dw2, db2, dw3, db3 = backward_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, x, y)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, lr)
        if i%50 == 0:
            _, _, _, _, _, a3_dev = forward_prop(w1, b1, w2, b2, w3, b3, x_dev)
            print('Iteration', i, 'has an accuracy of', round(get_accuracy(a3_dev, y_dev)*100, 2), '%')
    return w1, b1, w2, b2, w3, b3



if __name__ == "__main__":
    w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, y_train, x_dev, y_dev, 0.13, 2001)
    
    '''Save the trained parameters'''
    print('Saving values...')
    np.savetxt("w1.csv", w1, delimiter=',')
    np.savetxt("b1.csv", b1.flatten(), delimiter=',')
    np.savetxt("w2.csv", w2, delimiter=',')
    np.savetxt("b2.csv", b2.flatten(), delimiter=',')
    np.savetxt("w3.csv", w3, delimiter=',')
    np.savetxt("b3.csv", b3.flatten(), delimiter=',')

    print('Script is finished! Now you can go to NN Visualizer to test the model by yourself!')

