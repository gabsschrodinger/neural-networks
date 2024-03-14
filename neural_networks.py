from activation_functions import ActivationFunction, ActivationFunctionType
from nodes import HiddenNode, InputNode, OutputNode


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation_function_type: ActivationFunctionType,
    ) -> None:
        self._activation_function_type_: ActivationFunctionType = (
            activation_function_type
        )
        self.deactivation_function = ActivationFunction(
            activation_function_type
        ).get_activation_function_derivative()

        # Create input, hidden, and output nodes
        self.input_nodes = [
            InputNode(activation_function_type) for _ in range(input_size)
        ]
        self.hidden_nodes = [
            HiddenNode(activation_function_type) for _ in range(hidden_size)
        ]
        self.output_nodes = [OutputNode(activation_function_type) for _ in range(output_size)]

        # Connect input, hidden, and output nodes
        for input_node in self.input_nodes:
            input_node.set_next_layer_nodes(self.hidden_nodes)

        for hidden_node in self.hidden_nodes:
            hidden_node.set_next_layer_nodes(self.output_nodes)

    def feedforward(self, inputs: list[float]) -> list[float]:
        # Validate that there is exactly one input for each input node
        if len(inputs) != len(self.input_nodes):
            raise ValueError("Invalid number of inputs")

        # Set the value of each input node to the corresponding input
        for i, input_node in enumerate(self.input_nodes):
            input_node.set_value(inputs[i])

        # Pass the input to the hidden nodes
        for hidden_node in self.hidden_nodes:
            hidden_node.calculate_value()

        # Pass the output of the hidden nodes to the output nodes
        for output_node in self.output_nodes:
            output_node.calculate_value()

        # Return final output of the network
        return [output_node.get_output() for output_node in self.output_nodes]
    
    def update_output_weights_and_biases(self, delta: list[float], learning_rate: float) -> None:
        for i, output_node in enumerate(self.output_nodes):
            output_node.bias += learning_rate * delta[i]
            for connection in output_node.previous_layer:
                connection.weight += learning_rate * delta[i] * connection.src_node.get_output()
        
    def backward(
        self,
        expected_output: list[float],
        learning_rate: float,
    ) -> None:
        output_error = [expected_output[i] - self.output_nodes[i].get_output() for i in range(len(expected_output))]
        output_delta = [output_error[i] * self.deactivation_function(self.output_nodes[i].get_output()) for i in range(len(output_error))]
        self.update_output_weights_and_biases(output_delta, learning_rate)

        for _, hidden_node in enumerate(self.hidden_nodes):
            hidden_error = sum([output_delta[j] * hidden_node.next_layer[j].weight for j in range(len(self.output_nodes))])
            hidden_delta = hidden_error * self.deactivation_function(hidden_node.get_output())
            for connection in hidden_node.previous_layer:
                connection.weight += learning_rate * hidden_delta * connection.src_node.get_output()
            hidden_node.bias += learning_rate * hidden_delta

    def train(
        self,
        inputs: list[list[float]],
        expected_output: list[float],
        epochs: int,
        learning_rate: float,
    ) -> None:
        for _ in range(epochs):
            print(f"\n\nEpoch: {_}")
            for i in range(len(inputs)):
                self.feedforward(inputs[i])
                self.backward(expected_output[i], learning_rate)
