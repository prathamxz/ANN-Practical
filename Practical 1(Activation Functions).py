# ▶ How to Run the Program
# 1.	Install required libraries (if not already installed):
# pip install numpy matplotlib
# 2.	Save the program in a .py file, for example activation_functions.py.
# 3.	Run the file using:
# python activation_functions.py


import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def swish(x):
    return x * sigmoid(x)

# Generate input values
x = np.linspace(-10, 10, 1000)

# Compute activation values
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_swish = swish(x)

# Plot all activation functions
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color='blue')
plt.title("Sigmoid")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, y_tanh, label="Tanh", color='green')
plt.title("Tanh")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, y_relu, label="ReLU", color='red')
plt.title("ReLU")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, y_leaky_relu, label="Leaky ReLU", color='purple')
plt.title("Leaky ReLU")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, y_swish, label="Swish", color='orange')
plt.title("Swish")
plt.grid(True)

plt.tight_layout()
plt.suptitle("Activation Functions in Neural Networks", fontsize=16, y=1.02)
plt.show()