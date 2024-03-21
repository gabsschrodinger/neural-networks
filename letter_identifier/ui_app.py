import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

import json
import tkinter as tk

from neural_network.activation_functions import ActivationFunctionType
from neural_network.neural_networks import NeuralNetwork

class PixelArtApp:
    def __init__(self, master, size=25):
        self.master = master
        self.size = size
        self.canvas = tk.Canvas(master, width=size*12, height=size*15, bg='white')
        self.canvas.pack()
        self.create_grid()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.neural_network = NeuralNetwork(12*15, 50, 26, ActivationFunctionType.SIGMOID)
        self.neural_network_input = [0] * 12 * 15
        self.neural_network_training_file_path = "./neural_network_training_data.json"
        
        # self.train_neural_network()

        # self.guess_button = tk.Button(master, text="Guess", command=self.guess)
        # self.guess_button.pack()

        self.training_data_expected_output = tk.Entry(master)
        self.training_data_expected_output.pack()
        self.add_training_data_button = tk.Button(master, text="Add training data", command=self.save_training_data)
        self.add_training_data_button.pack()

        self._alphabet_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def guess(self):
        output = self.neural_network.feedforward(self.neural_network_input)
        letter_index = output.index(max(output))
        letter = self._alphabet_[letter_index]
        print(letter)

    def fill_network_input(self, x, y):
        self.neural_network_input[y * 12 + x] = 1

    def _map_expected_output_(self, expected_output: str) -> list[float]:
        neutral_output = [0.0] * 26
        letter_index = self._alphabet_.index(expected_output)
        neutral_output[letter_index] = 1.0
        return neutral_output
    
    def train_neural_network(self):
        with open(self.neural_network_training_file_path, "r") as file:
            training_data = json.load(file).get("training_data", [])
        
        inputs = [data.get("input") for data in training_data]
        outputs = [data.get("output") for data in training_data]
        self.neural_network.train(inputs, outputs, 20, 0.1)

    def save_training_data(self):
        expected_output = self.training_data_expected_output.get().upper()
        if expected_output == "" or expected_output not in self._alphabet_:
            return
        
        expected_output = self._map_expected_output_(expected_output)

        with open(self.neural_network_training_file_path, "r+") as file:
            current_data = json.load(file)
            current_training_data = current_data.get("training_data", [])
            print(current_training_data)
            current_training_data.append({"input": self.neural_network_input, "output": expected_output})
            current_data["training_data"] = current_training_data
            file.seek(0)
            json.dump(current_data, file)
            file.truncate()

    def create_grid(self):
        for i in range(0, self.size*20, self.size):
            self.canvas.create_line(i, 0, i, self.size*15, fill='gray')
            self.canvas.create_line(0, i, self.size*15, i, fill='gray')

    def paint(self, event):
        cell_x = event.x // self.size
        cell_y = event.y // self.size
        if 0 <= cell_x < 12 and 0 <= cell_y < 15:
            x = cell_x * self.size
            y = cell_y * self.size
            self.fill_network_input(cell_x, cell_y)
            self.canvas.create_rectangle(x, y, x+self.size, y+self.size, fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.neural_network_input = [0] * 12 * 15
        self.create_grid()

def main():
    root = tk.Tk()
    root.title("Pixel Art App")
    PixelArtApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()