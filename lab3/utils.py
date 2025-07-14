from math import exp

def relu(vector):
    return [max(0, x) for x in vector]

def relu_deriv(vector):
    return [1 if x > 0 else 0 for x in vector]

def sigmoid(vector):
    return [1 / (1 + exp(-x)) for x in vector]

def sigmoid_deriv(vector):
    sigmoid_vector = sigmoid(vector)
    return [x * (1 - x) for x in sigmoid_vector]

def vectors_tensor_product(v1, v2):
    return [[v1[i] * v2[j] for j in range(len(v2))] for i in range(len(v1))]

def transpose_matrix(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
