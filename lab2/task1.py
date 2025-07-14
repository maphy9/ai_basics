from ..lab1.task3 import neural_network

def next_epoc(input, weights, goal, alpha):
    output = neural_network(input, weights, 0)
    n = len(weights)
    m = len(input)
    tmp = [output[i] - goal[i] for i in range(n)]
    delta = [[2 * alpha / n * tmp[i] * input[j] for j in range(m)] for i in range(n)]
    
    new_weights = [[0 for i in range(m)] for j in range(n)]
    for i in range(n):
        for j in range(m):
            new_weights[i][j] = weights[i][j] - delta[i][j]
    return new_weights

def error(input, weights, goal):
    output = neural_network(input, weights, 0)
    n = len(weights)
    tmp = [(output[i] - goal[i]) ** 2 for i in range(n)]
    return sum(tmp) / n

if __name__ == '__main__':
    input = [2]
    weights = [[0.5]]
    goal = [0.8]
    alpha = 0.1

    for i in range(5):
        print(f'Epoc {i + 1}')
        print(f'error={error(input, weights, goal)}')
        print(f'prediction={neural_network(input, weights, 0)}')
        weights = next_epoc(input, weights, goal, alpha)
        print()
