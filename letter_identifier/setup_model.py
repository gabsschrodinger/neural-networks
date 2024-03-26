import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

import json
from neural_network.activation_functions import ActivationFunctionType
from neural_network.neural_networks import NeuralNetwork


model = NeuralNetwork(12 * 15, 300, 26, ActivationFunctionType.SIGMOID)

if os.path.exists("./models/letter_identifier_model.json"):
    model = NeuralNetwork.load_model("letter_identifier_model.json")

with open("./letter_identifier/training_data.json", "r") as file:
    training_data = json.load(file).get("training_data", [])

inputs = [data.get("input") for data in training_data]
expected_output = [data.get("output") for data in training_data]

model.train(inputs, expected_output, epochs=30, learning_rate=0.1)

model.save_model("letter_identifier_model.json")
