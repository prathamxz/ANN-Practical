#  How to Implement It:
# 	1.	Install dependencies (if not already): pip install scikit-learn matplotlib



import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target.reshape(-1, 1)  # reshape to column vector

# One-hot encode target
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ReLU and its derivative
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Loss function
def cross_entropy(pred, actual):
    return -np.sum(actual * np.log(pred + 1e-9)) / actual.shape[0]

# Accuracy
def accuracy(pred, actual):
    return np.mean(np.argmax(pred, axis=1) == np.argmax(actual, axis=1))

# Network architecture
input_dim = X.shape[1]
hidden_dim = 100
output_dim = Y.shape[1]
lr = 0.01
epochs = 500

# Weight initialization
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

losses = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Loss and accuracy
    loss = cross_entropy(A2, Y_train)
    acc = accuracy(A2, Y_train)
    losses.append(loss)

    # Backward pass
    dZ2 = A2 - Y_train
    dW2 = np.dot(A1.T, dZ2) / X_train.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X_train.shape[0]

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1) / X_train.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X_train.shape[0]

    # Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Final evaluation
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

test_acc = accuracy(A2_test, Y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plotting the loss curve
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()