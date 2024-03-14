import numpy as np
from activation_functions import ActivationFunctionType
from neural_networks import NeuralNetwork


inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
expected_output = [[0.0], [1.0], [1.0], [0.0]]

# inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
# expected_output = [[1.0], [0.0], [0.0], [1.0]]

input_size = 2
hidden_size = 100
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size, ActivationFunctionType.SIGMOID)

nn.train(inputs, expected_output, epochs=10000, learning_rate=0.1)

for i in range(len(inputs)):
    print(
        f"Input: {inputs[i]}, Expected: {expected_output[i]}, Output: {nn.feedforward(inputs[i])}"
    )
