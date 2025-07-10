import numpy as np
from keras.layers import Dense
from keras.models import Sequential

def deep_neural_network(input, weights_array):
    model = Sequential()
    
    for weights in weights_array:
        model.add(Dense(len(weights), weights=[np.transpose(np.array(weights))], use_bias=False))
    return model.predict(np.array([input]))


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
