from Framework.RegressionFramework import NeuralNetwork
from Framework.predict import predict_regression as predict
from Framework.normalize import normalize
from Data_Creation import regression_data


def run():
    # getting data
    x_train, y_train = regression_data.data(values=1000)

    # normalizing data
    # x_train = normalize(x_train)
    offset = 0
    factor = 1

    # instantiating object
    network = NeuralNetwork(x_train, y_train, learning_rate=0.06, num_layers=4)

    # displaying base information
    network.displayNN()

    # running our network
    for i in range(3):
        network.train()  # training our network

        # # display functions
        network.displayNN_forward(i)
        network.displayNN_backward(i)
        network.displayNN_loss(i)

        network.resetloss()  # resetting our loss after each iteration

    print("\n\n")

    # predicting our data
    predict(network, offset, factor)
