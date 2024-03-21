from neural_network.neural_networks import NeuralNetwork

def _output_handler_(output: list[float]) -> str:
    max_output = max(output)
    if max_output < 0.5:
        raise ValueError("No letter identified")

    _alphabet_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_index = output.index(max_output)
    return _alphabet_[letter_index]

def get_letter_identifier_model() -> NeuralNetwork:
    model = NeuralNetwork.load_model("letter_identifier_model.json")
    model.set_output_handler(_output_handler_)
    return model
