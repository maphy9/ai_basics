from neural_network import neural_network
from utils import *

def change_weights(outputs, weights_array, goals, alpha):
    output_layer_delta = [
        2 / len(weights_array[-1]) * (output - goal)
        for output, goal in zip(outputs[-1], goals)
    ]
    weights_deltas = [weights for weights in weights_array]
    output_layer_weights_delta = vectors_tensor_product(output_layer_delta, outputs[-2])
    weights_deltas[-1] = output_layer_weights_delta

    layers_count = len(outputs) - 1
    for l in range(layers_count - 1, 0, -1):
        output_layer_weights = weights_array[l]
        hidden_layer_output = outputs[l]

        transposed_output_layer_weights = transpose_matrix(output_layer_weights)
        n = len(transposed_output_layer_weights)
        m = len(output_layer_delta)
        hidden_layer_delta = [0 for i in range(n)]
        for i in range(n):
            for j in range(m):
                hidden_layer_delta[i] += transposed_output_layer_weights[i][j] * output_layer_delta[j]
        
        hidden_layer_output_relu_deriv = relu_deriv(hidden_layer_output)
        hidden_layer_delta = [hld * hlord for hld, hlord in zip(hidden_layer_delta, hidden_layer_output_relu_deriv)]

        input = outputs[l - 1]
        hidden_layer_weights_delta = vectors_tensor_product(hidden_layer_delta, input)
        weights_deltas[l - 1] = hidden_layer_weights_delta
        output_layer_delta = hidden_layer_delta

    for weights, weights_delta in zip(weights_array, weights_deltas):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = weights[i][j] - alpha * weights_delta[i][j]


def deep_neural_network(input, weights_array, goals, alpha):
    outputs = [input]
    for i, weights in enumerate(weights_array):
        output = neural_network(outputs[-1], weights, 0)
        if i < len(weights_array) - 1:
            output = relu(output)
        outputs.append(output)

    change_weights(outputs, weights_array, goals, alpha)

    return outputs[-1]


if __name__ == '__main__':
    inputs = [
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2],
    ]

    goals = [
        [0.1, 1.0, 0.1],
        [0.5, 0.2, -0.5],
        [0.1, 0.3, 0.2],
        [0.7, 0.6, 0.2],
    ]

    alpha = 0.01

    weights_array = [
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1]
        ],
        [
            [0.7, 0.9, -0.4, 0.8, 0.1],
            [0.8, 0.5, 0.3, 0.1, 0.0],
            [-0.3, 0.9, 0.3, 0.1, -0.2]
        ]
    ]
    
    for i in range(50):
        print(f'epoch #{i + 1}')
        for input, goal in zip(inputs, goals):
            res = deep_neural_network(input, weights_array, goal, alpha)
            print(res)
