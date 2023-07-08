from Framework.ClassificationFramework import NeuralNetwork
from Framework.predict import predict_classification as predict
from Framework.normalize import normalize
from Data_Creation import odd_even_data

def run():
    # getting data
    x_train, y_train = odd_even_data.data()

    # normalizing data
    x_train, offset, factor = normalize(x_train)

    # instantiating object
    network = NeuralNetwork(x_train, y_train, learning_rate=0.01, num_layers=3)

    # displaying base information
    network.displayNN()

    # running our network
    for i in range(10):
        network.train()  # training our network

        # # display functions
        network.displayNN_forward(i)
        # network.displayNN_backward(i)
        network.displayNN_loss(i)

        network.resetloss()  # resetting our loss after each iteration

    print("\n\n")

    # predicting our data
    predict(network, offset, factor)
