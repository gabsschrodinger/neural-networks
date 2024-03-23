import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

from neural_network.neural_networks import NeuralNetwork


def _output_handler_(output: list[float]) -> str:
    _alphabet_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_output = max(output)
    print(max_output)

    if max_output < 0.75:
        return f"Unknown letter. Confidence: {max_output}. First guess: {_alphabet_[output.index(max_output)]}"

    letter_index = output.index(max_output)
    return _alphabet_[letter_index]


def get_letter_identifier_model() -> NeuralNetwork:
    model = NeuralNetwork.load_model("letter_identifier_model.json")
    model.set_output_handler(_output_handler_)
    return model
