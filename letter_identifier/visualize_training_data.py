import json
import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

import tkinter as tk


class TrainingDataVisualizer:
    def __init__(self, master: tk.Tk, size: int = 25) -> None:
        self.master = master
        self.size = size
        self.canvas = tk.Canvas(master, width=size * 12, height=size * 15, bg="white")
        self.canvas.pack()
        self.create_grid()
        self.training_data = self.load_training_data()

        self.letter_label = tk.Label(master)
        self.letter_label.pack()

        self.current_training_data_index = 0
        self.display_training_data()
        self.next_letter_button = tk.Button(
            master, text="Next", command=self.next_training_data
        )
        self.next_letter_button.pack()

        self.previous_letter_button = tk.Button(
            master, text="Previous", command=self.previous_training_data
        )

        self.previous_letter_button.pack()

    def previous_training_data(self) -> None:
        self.current_training_data_index -= 1
        self.display_training_data()

    def next_training_data(self) -> None:
        self.current_training_data_index += 1
        self.display_training_data()

    def create_grid(self) -> None:
        for i in range(12):
            for j in range(15):
                x0, y0 = i * self.size, j * self.size
                x1, y1 = x0 + self.size, y0 + self.size
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="black", fill="white"
                )

    def load_training_data(self) -> list[dict]:
        file_path = "letter_identifier/training_data.json"
        with open(file_path, "r") as file:
            training_data = json.load(file).get("training_data", [])

        return training_data
    
    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.create_grid()

    def display_training_data(self) -> None:
        if self.current_training_data_index < 0:
            self.current_training_data_index = 0
        if self.current_training_data_index >= len(self.training_data):
            self.current_training_data_index = len(self.training_data) - 1

        self.clear_canvas()
        training_data = self.training_data[self.current_training_data_index]
        for i in range(12):
            for j in range(15):
                x0, y0 = i * self.size, j * self.size
                x1, y1 = x0 + self.size, y0 + self.size
                fill_color = (
                    "black" if training_data["input"][j * 12 + i] == 1 else "white"
                )
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color)

        letter = self._map_output(training_data["output"])
        self.letter_label["text"] = f"Letter: {letter}"
    
    def _map_output(self, output: list[float]) -> str:
        highest_value = max(output)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return alphabet[output.index(highest_value)]


def main():
    root = tk.Tk()
    root.title("Training Data Visualizer")

    TrainingDataVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
