import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

from neural_network.activation_functions import ActivationFunctionType
from neural_network.neural_networks import NeuralNetwork


inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
expected_output = [[0.0], [1.0], [1.0], [0.0]]

input_size = 2
hidden_size = 50
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size, ActivationFunctionType.SIGMOID)

nn.train(inputs, expected_output, epochs=10000, learning_rate=0.1)

nn.save_model("playground_model.json")

for i in range(len(inputs)):
    print(
        f"Input: {inputs[i]}, Expected: {expected_output[i]}, Output: {nn.feedforward(inputs[i])}"
    )

nn2 = NeuralNetwork.load_model("playground_model")

for i in range(len(inputs)):
    print(
        f"Input: {inputs[i]}, Expected: {expected_output[i]}, Output: {nn2.feedforward(inputs[i])}"
    )