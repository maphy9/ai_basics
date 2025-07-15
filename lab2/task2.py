from neural_network import NeuralNetwork
from task1 import error, next_epoc

def train_neural_network(nn, inputs, goals, alpha):
    # One layer neural network only
    err = 0
    for input, goal in zip(inputs, goals):
        weights = nn.weights_array[0]
        new_weights = next_epoc(input, weights, goal, alpha)

        nn.weights_array = [new_weights]
        err += error(input, new_weights, goal)
    return err


if __name__ == '__main__':
    inputs = [
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2]
    ]

    weights = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ]

    goals = [
        [0.1, 1.0, 0.1, 0.0, -0.1],
        [0.5, 0.2, -0.5, 0.3, 0.7],
        [0.1, 0.3, 0.2, 0.9, 0.1],
        [0.7, 0.6, 0.2, -0.1, 0.8]
    ]

    alpha = 0.01

    nn = NeuralNetwork(weights)

    for i in range(1000):
        err = train_neural_network(nn, inputs, goals, alpha)
        print(f'{i + 1}) {err}')
