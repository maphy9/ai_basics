from neural_network import neural_network
from utils import relu

def deep_neural_network(input, weights_array):
    output = input
    for i, weights in enumerate(weights_array):
        output = neural_network(output, weights, 0)
        if i < len(weights_array) - 1:
            output = relu(output)
    return output


if __name__ == '__main__':
    inputs = [
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2],
    ]

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
    
    for input in inputs:
        print(deep_neural_network(input, weights_array))
