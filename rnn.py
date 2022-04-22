
import numpy as np


def sigmoid(x):
    return 1.0 / (1+np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


time_step = 20
input_features = 10
output_features = 20
inputs = np.random.random((time_step, input_features))
h_state = np.random.random(output_features)
h_list = []


def generate_params(x_size, h_size, b_size):
    x = np.random.random(x_size)
    h = np.random.random(h_size)
    b = np.random.random(b_size)
    return x, h, b


x, h, b = generate_params((input_features, output_features),
                          (output_features, output_features),
                          (1, output_features))

for i in range(time_step):
    h_state = sigmoid(np.dot(inputs[i], x) + np.dot(h_state, h) + b)
    h_list.append(h_state)

print('The final output:')
print(h_state)