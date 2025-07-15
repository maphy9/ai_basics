from testing_set import inputs as testing_inputs, goals as testing_goals
from training_set import inputs as training_inputs, goals as training_goals
from task3 import Layer, NeuralNetwork

input_layer = Layer(4, 3, weight_min_value=0, weight_max_value=1, activation_function='relu')
nn = NeuralNetwork(input_layer)

i = 1
number_of_inputs = len(training_inputs)
number_of_iterations = 20
for _ in range(number_of_iterations):
    for input, expected_output in zip(training_inputs, training_goals):
        nn.fit(input, expected_output, alpha=0.01)
        print(' ' * 24, end='\r')
        print(f'{round(i / (number_of_inputs * number_of_iterations) * 100, 2)}%', end='\r')
        i += 1
print()
nn.save_weights('colors.json')

i = 1
correct_count = 0
for input, expected_output in zip(testing_inputs, testing_goals):
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
