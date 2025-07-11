from random import uniform
from json import load

def neuron(input, weights, bias):
    if len(input) != len(weights):
        raise Exception('Bad vector dimensions')

    output = bias
    for i, x in enumerate(input):
        w = weights[i]
        output += w * x

    return output


def neural_network(input, weights, bias):
    return [neuron(input, weights[i], bias) for i in range(len(weights))]


def deep_neural_network(input, weights_array):
    output = input
    for weights in weights_array:
        output = neural_network(output, weights, 0)
    return output


class NeuralNetwork:

    def __init__(self, first_layer=[]):
        self.weights_array = [first_layer]

    
    def add_layer(self, n, weight_min_value=-1, weight_max_value=1):
        m = len(self.weights_array[-1])

        new_layer = [[uniform(weight_min_value, weight_max_value) for i in range(m)] for j in range(n)] 
        
        self.weights_array.append(new_layer)

    
    def predict(self, input):
        return deep_neural_network(input, self.weights_array)
    
    @staticmethod
    def load_weights(filename):
        f = open(filename)
        content = load(f)
        f.close()

        weights_array = content['weights_array']
        neural_network = NeuralNetwork()
        neural_network.weights_array = weights_array
        return neural_network

