from task2 import train_neural_network
from neural_network import NeuralNetwork
from testing_set import inputs as testing_inputs, goals as testing_goals
from training_set import inputs as training_inputs, goals as training_goals
from random import uniform

if __name__ == '__main__':
    alpha = 0.01
    
    weights = [[uniform(0, 1) for i in range(3)] for j in range(4)]
    nn = NeuralNetwork(weights)

    for i in range(1000):
        err = train_neural_network(nn, training_inputs, training_goals, alpha)
        print(f'{i}) {err}')
    
    correct_count = 0
    for i, input in enumerate(testing_inputs):
        output = nn.predict(input)
        
        max_val = 0
        max_index = 0
        for j, val in enumerate(output):
            if val > max_val:
                max_val = val
                max_index = j
        goal = testing_goals[i]
        if goal[max_index] == 1:
            print(f'{i}) CORRECT')
            correct_count += 1
        else:
            print(f'{i}) INCORRECT')

    print(f'Result: {correct_count / len(testing_goals)}')
