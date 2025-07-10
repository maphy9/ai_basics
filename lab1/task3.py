from task2 import neural_network

def deep_neural_network(input, weights_array):
    output = input
    for weights in weights_array:
        output = neural_network(output, weights, 0)
    return output


if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
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
    print(deep_neural_network(input, weights_array))
