import math
import numpy as np


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))


time_steps = 10
input_features = 16
output_features = 32

input = np.random.random((time_steps, input_features))  # (10, 32)
c_state = np.zeros(shape=output_features,)
h_state = c_state  # 初始为0


def generate_params_for_lstm_cell(x_size, h_size, bias_size):
    """
    :param x_size:
    :param h_size:
    :param bias_size:
    :return:
    """
    w = np.random.random(x_size)
    u = np.random.random(h_size)
    b = np.random.random(bias_size)
    return w, u, b


forget_gate_w, forget_gate_u, forget_gate_b = generate_params_for_lstm_cell((input_features, output_features),
                                                                            (output_features, output_features),
                                                                            (1, output_features))


input_gate_w, input_gate_u, input_gate_b = generate_params_for_lstm_cell((input_features, output_features),
                                                                         (output_features, output_features),
                                                                         (1, output_features))

output_gate_w, output_gate_u, output_gate_b = generate_params_for_lstm_cell((input_features, output_features),
                                                                            (output_features, output_features),
                                                                            (1, output_features))

tanh_w, tanh_u, tanh_b = generate_params_for_lstm_cell((input_features, output_features),
                                                       (output_features, output_features),
                                                       (1, output_features))

h_list = []
for i in range(time_steps):
    # 遗忘门
    forget_gate = sigmoid(np.dot(h_state, forget_gate_u) + np.dot(input[i], forget_gate_w) + forget_gate_b)
    input_gate = sigmoid(np.dot(h_state, input_gate_u) + np.dot(input[i], forget_gate_w) + input_gate_b)
    output_gate = sigmoid(np.dot(h_state, output_gate_u) + np.dot(input[i], forget_gate_w) + output_gate_b)
    tanh_gate = np.tanh(np.dot(h_state, tanh_u) + np.dot(input[i], tanh_w) + tanh_b)
    c_state = c_state * forget_gate + input_gate * tanh_gate
    h_state = np.tanh(c_state) * output_gate
    h_list.append(h_state)


print('最后一个cell的输出')
print(h_state)
print('每一层cell的输出')
print(h_list)
