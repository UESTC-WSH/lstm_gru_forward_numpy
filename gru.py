import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


time_steps = 10
input_features = 16
output_features = 32


inputs = np.random.random((time_steps, input_features))
h_state = np.random.random(output_features)
h_list = []


def generate_params(x_size, h_size, b_size):
    w = np.random.random(x_size)
    h = np.random.random(h_size)
    b = np.random.random(b_size)
    return w, h, b


update_gate_w, update_gate_h, update_gate_b = generate_params((input_features, output_features),
                                                              (output_features, output_features),
                                                              (1, output_features))

reset_gate_w, reset_gate_h, reset_gate_b = generate_params((input_features, output_features),
                                                              (output_features, output_features),
                                                              (1, output_features))

_, h_w, h_b = generate_params((input_features, output_features),
                              (output_features, output_features),
                              (1, output_features))
x_w, _, x_b = generate_params((input_features, output_features),
                              (output_features, output_features),
                              (1, output_features))

for i in range(time_steps):
    update_gate = sigmoid(np.dot(h_state, update_gate_h) + np.dot(inputs[i], update_gate_w) + update_gate_b)
    reset_gate = sigmoid(np.dot(h_state, reset_gate_h) + np.dot(inputs[i], reset_gate_w) + reset_gate_b)
    h_ = np.dot(h_state, h_w) + h_b
    x_ = np.dot(inputs[i], x_w) + x_b
    h_state = h_state * update_gate + (1-update_gate) * tanh(reset_gate * h_ + x_)
    h_list.append(h_state)

print('最终输出')
print(h_state)
print('隐藏层')
print(h_list)