from random import uniform
from neural_network import deep_neural_network, neural_network
from json import load, dump
from utils import *

class Layer:
    def __init__(self, m=0, n=0, weight_min_value=0., weight_max_value=0., activation_function=None):
        weights = [[uniform(weight_min_value, weight_max_value) for i in range(n)] for j in range(m)] 
        self.weights = weights

        self.activation_function_str = activation_function
        self.activation_function = None
        self.activation_function_derivative = None
        if activation_function == 'relu':
            self.activation_function = relu
            self.activation_function_derivative = relu_deriv
        elif activation_function == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_function_derivative = sigmoid_deriv


    def json(self):
        return {
            'weights': self.weights,
            'activation_function': self.activation_function_str,
        }


    def get_output_size(self):
        return len(self.weights)

class NeuralNetwork:

    def __init__(self, first_layer=None):
        if not first_layer:
            first_layer = Layer(3, 3)
        self.layers = [first_layer]

    
    def add_layer(self, m, weight_min_value=-1., weight_max_value=1., activation_function=None):
        n = self.layers[-1].get_output_size()
        new_layer = Layer(m, n, weight_min_value, weight_max_value, activation_function)
        self.layers.append(new_layer)

    
    def predict(self, input):
        output = input
        for layer in self.layers:
            weights = layer.weights
            activation_function = layer.activation_function
            output = neural_network(output, weights, 0)
            if activation_function:
                output = activation_function(output)
        return output


    def __change_weights(self, outputs, expected_outputs, alpha):
        layers = self.layers

        output_layer_delta = [
            2 / layers[-1].get_output_size() * (output - expected_output)
            for output, expected_output in zip(outputs[-1], expected_outputs)
        ]
        weights_deltas = [[[]] for layer in layers]
        output_layer_weights_delta = vectors_tensor_product(output_layer_delta, outputs[-2])
        weights_deltas[-1] = output_layer_weights_delta

        layers_count = len(outputs) - 1
        for l in range(layers_count - 1, 0, -1):
            layer = layers[l]
            output_layer_weights = layer.weights
            activation_function_derivative = layer.activation_function_derivative
            hidden_layer_output = outputs[l]

            transposed_output_layer_weights = transpose_matrix(output_layer_weights)
            n = len(transposed_output_layer_weights)
            m = len(output_layer_delta)
            hidden_layer_delta = [0 for i in range(n)]
            for i in range(n):
                for j in range(m):
                    hidden_layer_delta[i] += transposed_output_layer_weights[i][j] * output_layer_delta[j]
            
            if activation_function_derivative:
                hidden_layer_output_deriv = activation_function_derivative(hidden_layer_output)
                hidden_layer_delta = [hld * hlord for hld, hlord in zip(hidden_layer_delta, hidden_layer_output_deriv)]

            input = outputs[l - 1]
            hidden_layer_weights_delta = vectors_tensor_product(hidden_layer_delta, input)
            weights_deltas[l - 1] = hidden_layer_weights_delta
            output_layer_delta = hidden_layer_delta

        for layer, weights_delta in zip(layers, weights_deltas):
            weights = layer.weights
            for i in range(len(weights)):
                for j in range(len(weights[i])):
                    weights[i][j] = weights[i][j] - alpha * weights_delta[i][j]


    def fit(self, input, expected_outputs, alpha=0.01):
        outputs = [input]
        for i, layer in enumerate(self.layers):
            weights = layer.weights
            activation_function = layer.activation_function
            output = neural_network(outputs[-1], weights, 0)
            if activation_function:
                output = activation_function(output)
            outputs.append(output)

        self.__change_weights(outputs, expected_outputs, alpha)
    

    @staticmethod
    def load_weights(filename):
        f = open(filename)
        content = load(f)
        f.close()

        layers = content['layers']
        neural_network = NeuralNetwork()
        neural_network.layers = [Layer(activation_function=layer['activation_function']) for layer in layers]
        for layer, nn_layer in zip(layers, neural_network.layers):
            nn_layer.weights = layer['weights']
        return neural_network

    
    def save_weights(self, filename):
        data = {
            'layers': [layer.json() for layer in self.layers]
        }
        with open(filename, 'w') as f:
            dump(data, f)



if __name__ == '__main__':
    def read_inputs(filename):
        inputs = []
        with open(filename, 'rb') as f:
            magic_number = int.from_bytes(f.read(4))
            if magic_number != 2051:
                raise Exception('Bad magic number')
            number_of_data = int.from_bytes(f.read(4))
            image_rows = int.from_bytes(f.read(4))
            image_cols = int.from_bytes(f.read(4))
            for k in range(number_of_data):
                image = [0 for i in range(image_rows * image_cols)]
                for i in range(image_rows * image_cols):
                    b = int.from_bytes(f.read(1))
                    image[i] = b / 255
                inputs.append(image)
        return inputs


    def read_outputs(filename):
        outputs = []
        with open(filename, 'rb') as f:
            magic_number = int.from_bytes(f.read(4))
            if magic_number != 2049:
                raise Exception('Bad magic number')
            number_of_data = int.from_bytes(f.read(4))
            for k in range(number_of_data):
                label = int.from_bytes(f.read(1))
                outputs.append([0 if i != label else 1 for i in range(10)])
        return outputs

    input_layer = Layer(40, 784, weight_min_value=-0.1, weight_max_value=0.1, activation_function='relu')
    nn = NeuralNetwork(input_layer)
    nn.add_layer(10, weight_min_value=-0.1, weight_max_value=0.1)

    training_inputs = read_inputs('datasets/train-images.idx3-ubyte')
    print('read training images')
    training_outputs = read_outputs('datasets/train-labels.idx1-ubyte')
    print('read training labels')
    i = 1
    for input, expected_outputs in zip(training_inputs, training_outputs):
        nn.fit(input, expected_outputs, alpha=0.005)
        print(' ' * 100 , end='\r')
        print(f'{round(i / len(training_inputs) * 100, 2)}%', end='\r')
        i += 1
    print()
    print('neural network fitting finished')
    nn.save_weights('results.json')
    # nn = NeuralNetwork.load_weights('results.json')

    testing_inputs = read_inputs('datasets/t10k-images.idx3-ubyte')
    print('read testing images')
    testing_outputs = read_outputs('datasets/t10k-labels.idx1-ubyte')
    print('read testing labels')
    i = 1
    correct_count = 0
    for input, expected_outputs in zip(testing_inputs, testing_outputs):
        res = nn.predict(input)
        max_index = 0
        for j in range(len(res)):
            if res[j] > res[max_index]:
                max_index = j
        print(' ' * 100 , end='\r')
        if expected_outputs[max_index] == 1:
            print(f'{i}) Correct', end='\r')
            correct_count += 1
        else:
            print(f'{i}) Incorrect', end='\r')
        i += 1
    print()
    print(f'Accuracy = {correct_count / len(testing_inputs)}')

