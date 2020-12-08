# imports
import numpy as np

# Code for 3 layer NN network from scratch without
# using any frameworks

'''
    Neural Networks consist of the following components :
    - An input layer [x]
    - An arbitrary amount of hidden layers
    - An output layer [ŷ]
    - A set of weights and biases between each layer, W and b
    - A choice of activation function for each hidden layer,
    - Loss Function ( L(x) = ( y - ŷ )² ) [Root Mean Square]
    - Learning Rate (alpha)
    - Optimization Algorithm (Gradient Descent)

    Naturally, the right values for the weights and biases 
    determines the strength of the predictions. The process 
    of fine-tuning the weights and biases from the input data 
    is known as training the Neural Network.

    Neural Networks have 2 steps in every iteration of training :
    - Calculating the predicted output ŷ, known as feedforward
    - Updating the weights and biases, known as backpropagation

    However, we still need a way to evaluate how far off are our 
    predictions. The Loss Function allows us to do exactly that.

        loss function :                L(x) = Σ ( y - ŷ )^2
        cost function :                L(x) =   ( y - ŷ )^2
        activation function :          σ(z) = 1 / ( 1 + e^(-z) )

    After measuring the error of our prediction (loss), we need
    to find a way to propagate the error back, and to update our 
    weights and biases. In order to know the appropriate amount to 
    adjust the weights and biases by, we need to know the derivative 
    of the loss function with respect to the weights and biases.

    If we have the derivative, we can simply update the weights and 
    biases by increasing/reducing with it. This is known as 
    gradient descent.

        gradient descent updations :
                    ∂z = α * ∂(J) / ∂z
                    W += ∂z * ∂(z) / ∂w
                    b += ∂z * ∂(z) / ∂b

    This Algorithm below uses Stochastic gradient Descent to
    propogate, in stead of Batch or Mini-Batch Grad Descent.
'''

# nn class
class NeuralNetwork:

    # initializing variables
    def __init__(self, x, y, learning_rate=0.06, num_layers=2):
        # input array
        self.input = x

        # learning rate of gradient descent value
        self.alpha = learning_rate

        # number of layers of NN (not including input layer)
        self.num_layers = num_layers

        # creating array that hold the number of nodes in each layer
        self.num_nodes = np.random.randint(
            2, high=10, size=num_layers-1).tolist()
        self.num_nodes.insert(0, 1)
        self.num_nodes.append(1)

        # setting weights of all layers
        for i in range(len(self.num_nodes)-1):
            # dynamically creating weights
            cmd = "self.w{} = np.random.randn(self.num_nodes[i+1], self.num_nodes[i])".format(
                i+1)
            exec(cmd)

        # setting biases of all layers
        for i in range(len(self.num_nodes)-1):
            # dynamically creating biases
            cmd = "self.b{} = np.random.randn(self.num_nodes[i+1], 1)".format(
                i+1)
            exec(cmd)

        # output array and it's shape
        self.y = y
        self.y_hat = np.random.rand(y.shape[0], y.shape[1])

        self.loss = 0

    # displaying basic information about the Neural Network
    def displayNN(self):
        print("\n==============================Neural Network Properties==============================")

        print("\nThe number of layers in this NN are : ", end="")
        print(self.num_layers)

        print("\nThe learning rate of this NN are : ", end="")
        print(self.alpha)

        print("\nThe number of nodes in each NN are : ", end="")
        print(self.num_nodes)

        print("\n\nThe weights of each layer of the NN is : \n")
        for i in range(self.num_layers):
            print("w{} : ".format(i+1))
            cmd = "print(self.w{})".format(i+1)
            exec(cmd)
            print("\n")

        print("The shape weight matrices of each layer are : ")
        for i in range(self.num_layers):
            size = []
            cmd = "size.append(self.w{}.shape)".format(i+1)
            exec(cmd)
            print("W{} shape : {}".format(i+1, size))

        print("\n==============================Neural Network Properties==============================\n")

    # displaying information about the Forward Propogation Step
    def displayNN_forward(self, step):
        print("\n==============================Forward Propogation Step : {}==============================\n".format(step+1))

        for i in range(self.num_layers+1):
            size = []
            cmd = "size.append(self.z{}.shape)".format(i)
            exec(cmd)
            print("\nLayer Z{} shape : {}".format(i, size), end="")

        for i in range(1, self.num_layers+1):
            size = []
            cmd = "size.append(self.a{}.shape)".format(i)
            exec(cmd)
            print("\nA{} shape : {}".format(i, size), end="")

        print("\n")

        for i in range(self.num_layers+1):
            print("Layer Z{} : ".format(i))
            cmd = "print(self.z{})".format(i)
            exec(cmd)
            print("\n")

        for i in range(1, self.num_layers+1):
            print("Activation A{} : ".format(i))
            cmd = "print(self.a{})".format(i)
            exec(cmd)
            print("\n")

        print("==============================Forward Propogation==============================\n")

    # displaying information about the Backward Propogation Step
    def displayNN_backward(self, step):
        print("\n==============================Backward Propogation Step : {}==============================\n".format(step+1))

        for i in range(1, self.num_layers+1):
            size = []
            cmd = "size.append(self.dz{}.shape)".format(i)
            exec(cmd)
            print("Diffrential Layer dZ{} shape : {}\n".format(i, size), end="")

        print("")

        for i in range(1, self.num_layers+1):
            size = []
            cmd = "size.append(self.dw{}.shape)".format(i)
            exec(cmd)
            print("dW{} shape : {}\n".format(i, size), end="")

        print("")

        for i in range(1, self.num_layers+1):
            print("Layer {} dZ : ".format(i))
            cmd = "print(self.dz{})".format(i)
            exec(cmd)
            print("")

        for i in range(1, self.num_layers+1):
            print("dW{} : ".format(i))
            cmd = "print(self.dw{})".format(i)
            exec(cmd)
            print("")

        print("==============================Backward Propogation==============================\n")

    # displaying information about the Loss of each Step
    def displayNN_loss(self, step):
        step += 1

        loss = self.loss/self.y.shape[0]

        print("==============================Loss of Step : {}=========================".format(step))

        print("loss = {}, loss_percent = {}%".format(
            round(loss, 4), round(loss*100, 3)))
        print("accuracy = {}, accuracy_percent = {}%".format(
            round(1-loss, 4), round((1-loss)*100, 3)))

        print("==============================Loss of Step==============================")

    # Calculating Loss
    def NN_loss(self, y_hat, y):
        loss = (y_hat - y)**2
        loss = loss[0]

        self.loss += loss

    # training neural net
    def train(self):

        # dynamically calculating layers and their respective z
        for i in range(len(self.input)):

            self.z0 = self.input[i].reshape([-1, 1])

            # forward step
            output = self.forwardprop()

            self.y_hat[i] = output

            # backward step
            self.backprop(self.y_hat[i], self.y[i])

            # loss calculation step
            self.NN_loss(self.y_hat[i], self.y[i])

    # Forward Propagation Logic
    def forwardprop(self):
        # dynamically calculating first layer
        exec("self.z1 = np.dot( self.w1, self.z0 ) + self.b1")

        # dynamically calculating the "a" of layer
        exec("self.a1 = self.sigmoid(self.z1)")

        # dynamically calculating all other layers
        for i in range(2, self.num_layers + 1):
            # dynamically calculating "z"
            cmd1 = "self.z{} = np.dot( self.w{}, self.a{} ) + self.b{}".format(
                i, i, i-1, i)

            # dynamically calculating the "a" of layers
            cmd2 = "self.a{} = self.sigmoid(self.z{})".format(i, i)

            # executing code
            exec(cmd1)
            exec(cmd2)

        # returning output
        temp = []
        exec("temp.append(self.a{}[0][0])".format(self.num_layers))
        return temp

    # activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # returning loss
    def ret_loss(self):
        loss_arr = []

        cmd = "loss_arr = (self.a{} - self.y)**2".format(self.num_layers)
        exec(cmd)

        return np.sum(loss_arr)/self.num_nodes[-1]

    # Backward Propagation Logic
    def backprop(self, y_hat, y):
        # using chain rule to chain rule to find derivative of the
        # loss function with respect to the last layer i.e. z

        j = self.num_layers

        # calculating last dz
        cmd = "self.dz{} = 2 * (y_hat - y) * self.d_sigmoid(self.z{})".format(j, j, j)
        exec(cmd)

        # calculating all other dzs
        for i in range(j-1, 0, -1):

            cmd = "self.dz{} = self.d_sigmoid(self.z{}) * np.dot( self.w{}.T , self.dz{} )".format(
                i, i, i+1, i+1)
            exec(cmd)

        # calculating and updating the weights
        for i in range(j, 1, -1):
            # creating dW
            cmd1 = "self.dw{} = np.dot( self.dz{}, np.transpose(self.a{}) )".format(
                i, i, i-1)

            # updating W using dW
            cmd2 = "self.w{} -= self.alpha*self.dw{}".format(i, i)

            exec(cmd1)
            exec(cmd2)
        
        # calculating and updating the first weight
        # creating dW
        cmd1 = "self.dw1 = np.dot( self.dz1, self.z0.T )"

        # updating W using dW
        cmd2 = "self.w1 -= self.alpha*self.dw1"

        # updating B using dz
        for i in range(j, 0, -1):
            cmd2 = "self.b{} -= self.alpha*self.dz{}".format(i, i)

        exec(cmd1)
        exec(cmd2)

    # activation function derrivative
    def d_sigmoid(self, z):
        eta = self.sigmoid(z)
        return eta * (1 - eta)

    # reset loss after each iteration
    def resetloss(self):
        self.loss = 0

    # Predicting input values
    def predict(self, num):

        # setting input as the first layer
        self.z0 = num

        # forward propagating and returning the result
        return self.forwardprop()[0]
