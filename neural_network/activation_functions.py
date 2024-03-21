import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

from enum import Enum

import numpy as np


class ActivationFunctionType(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    NONE = 4


class ActivationFunction:
    def __init__(self, activation_function_type: ActivationFunctionType) -> None:
        self._activation_function_type_: ActivationFunctionType = (
            activation_function_type
        )

    def sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def tanh(self, x: float) -> float:
        return np.tanh(x)

    def relu(self, x: float) -> float:
        return max(0, x)

    def no_activation(self, x: float) -> float:
        return x

    def sigmoid_derivative(self, x: float) -> float:
        return x * (1 - x)

    def tanh_derivative(self, x: float) -> float:
        return 1 - x**2

    def relu_derivative(self, x: float) -> float:
        return 1 if x > 0 else 0

    def get_activation_function(self):
        if self._activation_function_type_ == ActivationFunctionType.SIGMOID:
            return self.sigmoid
        elif self._activation_function_type_ == ActivationFunctionType.TANH:
            return self.tanh
        elif self._activation_function_type_ == ActivationFunctionType.RELU:
            return self.relu
        elif self._activation_function_type_ == ActivationFunctionType.NONE:
            return self.no_activation

    def get_activation_function_derivative(self):
        if self._activation_function_type_ == ActivationFunctionType.SIGMOID:
            return self.sigmoid_derivative
        elif self._activation_function_type_ == ActivationFunctionType.TANH:
            return self.tanh_derivative
        elif self._activation_function_type_ == ActivationFunctionType.RELU:
            return self.relu_derivative
        elif self._activation_function_type_ == ActivationFunctionType.NONE:
            return self.no_activation
