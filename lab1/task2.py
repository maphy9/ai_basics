from task1 import neuron

def neural_network(input, weights, bias):
    return [neuron(input, weights[i], bias) for i in range(len(weights))]

if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
    weights = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ]
    bias = 0
    print(neural_network(input, weights, bias))
