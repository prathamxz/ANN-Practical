# ▶ How to Implement	
# 1.	Save the program as bam_network.py 
# 2.	Run it using: python bam_network.py

import numpy as np

def sign(x):
    return np.where(x >= 0, 1, -1)

# Training data: two pairs
X = np.array([[1, -1], [-1, 1]])
Y = np.array([[1, -1, 1], [-1, 1, -1]])

# Initialize weight matrix
W = np.zeros((X.shape[1], Y.shape[1]))
for i in range(len(X)):
    W += np.outer(X[i], Y[i])

print("Weight Matrix W:")
print(W)

# Test recall from X → Y
print("\nRecall from X to Y:")
for i in range(len(X)):
    y_pred = sign(X[i] @ W)
    print(f"Input X{i+1}: {X[i]}, Recalled Y: {y_pred}")

# Test recall from Y → X
print("\nRecall from Y to X:")
for i in range(len(Y)):
    x_pred = sign(Y[i] @ W.T)
    print(f"Input Y{i+1}: {Y[i]}, Recalled X: {x_pred}")