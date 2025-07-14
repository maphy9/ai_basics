def neuron(input, weights, bias):
    output = bias
    for x, w in zip(input, weights):
        output += w * x

    return output
    

if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
    weights = [0.1, 0.1, -0.3]
    bias = 0
    print(neuron(input, weights, bias))
