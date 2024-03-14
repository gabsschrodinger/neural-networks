from enum import Enum
import numpy as np

from activation_functions import ActivationFunction, ActivationFunctionType


class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node:
    def __init__(
        self, type: NodeType, activation_function_type: ActivationFunctionType
    ) -> None:
        self.activation_function = ActivationFunction(activation_function_type).get_activation_function()
        self.type: NodeType = type
        self._value_: float = None
        self.bias:float = 0.0
        self.next_layer: list[NodeConnection] = []
        self.previous_layer: list[NodeConnection] = []

    def set_next_layer_nodes(self, nodes: list["Node"]) -> None:
        for node in nodes:
            node_connection = NodeConnection(self, node)
            self.next_layer.append(node_connection)
            node.previous_layer.append(node_connection)

    def get_output(self) -> float:
        if self.type == NodeType.INPUT:
            return self._value_
        else:
            return self.activation_function(self._value_)

    def calculate_value(self) -> None:
        product_from_previous_layer = sum(
            [
                connection.src_node.get_output() * connection.weight
                for connection in self.previous_layer
            ]
        )
        self._value_ = product_from_previous_layer + self.bias


class NodeConnection:
    def __init__(self, src_node: Node, tgt_node: Node) -> None:
        self.src_node = src_node
        self.tgt_node = tgt_node
        self.weight = np.random.standard_normal()


class HiddenNode(Node):
    def __init__(self, activation_function_type: ActivationFunctionType) -> None:
        super().__init__(NodeType.HIDDEN, activation_function_type)


class InputNode(Node):
    def __init__(self, activation_function_type: ActivationFunctionType) -> None:
        super().__init__(NodeType.INPUT, activation_function_type)

    def set_value(self, value: float) -> None:
        self._value_ = value


class OutputNode(Node):
    def __init__(self, activation_function_type: ActivationFunctionType) -> None:
        super().__init__(NodeType.OUTPUT, activation_function_type)
