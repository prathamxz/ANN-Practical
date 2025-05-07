import numpy as np

class HopfieldNetwork:
    def __init__(self):
        self.num_neurons = 0
        self.weights = None

    def train(self, patterns):
        self.num_neurons = patterns.shape[1]
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        # Hebbian learning rule
        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += np.dot(p, p.T)

        np.fill_diagonal(self.weights, 0)
        print("Weight matrix after training:\n", self.weights)

    def recall(self, pattern, steps=5):
        pattern = pattern.copy()
        print("\nRecalling pattern:")
        for step in range(steps):
            for i in range(self.num_neurons):
                raw = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw >= 0 else -1
        return pattern

# --- Define patterns to store ---
patterns = np.array([
    [1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1],
    [-1, -1, 1, 1, -1, -1]
])

# --- Initialize and train network ---
hopfield = HopfieldNetwork()
hopfield.train(patterns)

# --- Test recall with a noisy pattern ---
test_pattern = np.array([1, -1, 1, -1, -1, -1])  # noisy version of first pattern
output = hopfield.recall(test_pattern)
print("Recovered Pattern:", output)