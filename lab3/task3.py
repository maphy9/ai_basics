from random import uniform
from neural_network import neural_network
from json import load, dump
from utils import *

class Layer:
    def __init__(self, m=0, n=0, weight_min_value=-1., weight_max_value=1., activation_function=None):
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

    
    
    def fit(self, input, expected_output, alpha=0.01):
        # Step 1: calculate output for each layer
        layer_inputs = [[] for layer in self.layers]
        layer_inputs[0] = input
        layer_outputs = [[] for layer in self.layers]
        for layer_index in range(len(self.layers)):
            input = layer_inputs[layer_index]
            layer = self.layers[layer_index]
            weights = layer.weights
            output = [0 for i in range(len(weights))]
            for i in range(len(weights)):
                for j in range(len(input)):
                    output[i] += weights[i][j] * input[j]
            if layer.activation_function:
                output = layer.activation_function(output)
            if layer_index < len(self.layers) - 1:
                layer_inputs[layer_index + 1] = output
            layer_outputs[layer_index] = output

        # Step 2: calculate output layer delta
        output_layer_output = layer_outputs[-1]
        expected_output_difference = subtract_vectors(output_layer_output, expected_output)
        n = len(output_layer_output)
        output_layer_delta = multiply_vector(expected_output_difference, 2 / n)
        layer_deltas = [[] for layer in self.layers]
        layer_deltas[-1] = output_layer_delta

        # Step 3: calculate hidden layer deltas
        for layer_index in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            weights = next_layer.weights
            transposed_weights = transpose_matrix(weights)
            next_layer_delta = layer_deltas[layer_index + 1]
            layer_delta = multiply_matrix_vector(transposed_weights, next_layer_delta)
            if layer.activation_function_derivative:
                layer_output = layer_outputs[layer_index]
                layer_output_deriv = layer.activation_function_derivative(layer_output)
                layer_delta = multiply_vectors(layer_delta, layer_output_deriv)
            layer_deltas[layer_index] = layer_delta

        # Step 4: calculate layer weights deltas
        for layer_index in range(len(self.layers)):
            layer_delta = layer_deltas[layer_index]
            layer_input = layer_inputs[layer_index]
            layer_weights_delta = vectors_tensor_product(layer_delta, layer_input)
            layer_weights_delta = multiply_matrix(layer_weights_delta, alpha)
            old_weights = self.layers[layer_index].weights
            new_weights = subtract_matrixes(old_weights, layer_weights_delta)
            self.layers[layer_index].weights = new_weights
    

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

    # input_layer = Layer(40, 784, weight_min_value=-0.1, weight_max_value=0.1, activation_function='relu')
    # nn = NeuralNetwork(input_layer)
    # nn.add_layer(10, weight_min_value=-0.1, weight_max_value=0.1)

    nn = NeuralNetwork.load_weights('results.json')
    training_inputs = read_inputs('datasets/train-images.idx3-ubyte')
    print('read training images')
    training_outputs = read_outputs('datasets/train-labels.idx1-ubyte')
    print('read training labels')

    i = 1
    number_of_inputs = len(training_inputs)
    number_of_iterations = 2
    for _ in range(number_of_iterations):
        for input, expected_output in zip(training_inputs, training_outputs):
            nn.fit(input, expected_output, alpha=0.001)
            print(' ' * 24, end='\r')
            print(f'{round(i / (number_of_inputs * number_of_iterations) * 100, 2)}%', end='\r')
            i += 1
    print()
    print('neural network fitting finished')
    nn.save_weights('results.json')

    testing_inputs = read_inputs('datasets/t10k-images.idx3-ubyte')
    print('read testing images')
    testing_outputs = read_outputs('datasets/t10k-labels.idx1-ubyte')
    print('read testing labels')
    i = 1
    correct_count = 0
    for input, expected_output in zip(testing_inputs, testing_outputs):
        res = nn.predict(input)
        max_index = 0
        for j in range(len(res)):
            if res[j] > res[max_index]:
                max_index = j
        if expected_output[max_index] == 1:
            correct_count += 1
        print(' ' * 24, end='\r')
        print(f'{round(i * 100 / len(testing_inputs), 2)}\taccuracy: {round(correct_count * 100 / i, 2)}', end='\r')
        i += 1
    print()
    print(f'Accuracy = {correct_count / len(testing_inputs)}')

