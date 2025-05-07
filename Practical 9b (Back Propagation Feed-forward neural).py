# ▶ How to Implement
# 	1.	Save the file as backprop_nn.py
# 	2.	Run using: python backprop_nn.py


import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data: XOR inputs and outputs
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Network configuration
input_layer_size = 2
hidden_layer_size = 4
output_layer_size = 1

# Weight initialization
np.random.seed(1)
W1 = 2 * np.random.rand(input_layer_size, hidden_layer_size) - 1
b1 = np.zeros((1, hidden_layer_size))

W2 = 2 * np.random.rand(hidden_layer_size, output_layer_size) - 1
b2 = np.zeros((1, output_layer_size))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # --- Forward Propagation ---
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # Output

    # --- Backward Propagation ---
    error = y - a2
    d_output = error * sigmoid_derivative(a2)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(a1)

    # --- Update Weights and Biases ---
    W2 += a1.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# --- Final Output ---
print("\nFinal Output after Training:")
print(np.round(a2))