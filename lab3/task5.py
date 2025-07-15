import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from testing_set import inputs as testing_inputs, goals as testing_goals
from training_set import inputs as training_inputs, goals as training_goals
from random import uniform

model = Sequential()
weights1 = [np.array([[uniform(0, 1) for i in range(4)] for j in range(3)])]
weights2 = [np.array([[uniform(0, 1) for i in range(4)] for j in range(4)])]
model.add(Dense(4, weights=weights1, use_bias=False, activation='relu'))
model.add(Dense(4, weights=weights2))

opt = SGD(lr=0.01)
model.compile(opt, loss='mse')

for _ in range(15):
    for input, expected_output in zip(np.array(training_inputs), np.array(training_goals)):
        model.fit(np.array([input]), np.array([expected_output]))

correct_count = 0
for input, expected_output in zip(np.array(testing_inputs), np.array(testing_goals)):
    output = model.predict(np.array([input]))[0]
    max_val = 0
    max_index = 0
    for j, val in enumerate(output):
        if val > max_val:
            max_val = val
            max_index = j
    if expected_output[max_index] == 1:
        correct_count += 1

print(f'Result: {correct_count / len(testing_goals)}')
