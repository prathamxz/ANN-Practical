# ▶ How to Implement
# 	1.	Save the file as art1_network.py
# 	2.	Run the program: python art1_network.py

import numpy as np

class ART1:
    def __init__(self, input_size, vigilance=0.8):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []  # list of bottom-up weights (F1 → F2)

    def _match(self, input_vector, weight):
        # Check match with current weight
        match_score = np.sum(np.minimum(input_vector, weight)) / np.sum(input_vector)
        return match_score

    def _resonance(self, input_vector, weight):
        # Resonance if match is above vigilance
        return self._match(input_vector, weight) >= self.vigilance

    def train(self, data):
        for x in data:
            x = np.array(x)
            recognized = False
            for i, w in enumerate(self.weights):
                if self._resonance(x, w):
                    print(f"Pattern {x} → Category {i}")
                    self.weights[i] = np.minimum(w, x)  # update weights
                    recognized = True
                    break
            if not recognized:
                self.weights.append(x.copy())  # new category
                print(f"Pattern {x} → New Category {len(self.weights) - 1}")

# Example usage
if __name__ == "__main__":
    # Binary input patterns (each of length 6)
    patterns = [
        [1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ]

    art = ART1(input_size=6, vigilance=0.8)
    art.train(patterns)