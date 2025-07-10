from random import uniform
from task3 import deep_neural_network
from json import load

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


if __name__ == '__main__':
    neural_network1 = NeuralNetwork.load_weights('weights.json')
    input = [0.5, 0.75, 0.1]
    print(neural_network1.predict(input))

    first_layer = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ]
    neural_network2 = NeuralNetwork(first_layer)
    for i in range(10):
        neural_network2.add_layer(5 + i)
    print(neural_network2.predict(input))
