# ▶ How to Implement
# 	1.	Install dependencies (if not already): pip install numpy matplotlib scikit-learn
# 	2.	Save as perceptron_decision_boundary.py
# 	3.	Run it: python perceptron_decision_boundary.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

# Define OR gate data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1])  # OR logic

# Create and train perceptron
clf = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
clf.fit(X, y)

# Print weights and bias
print("Weights:", clf.coef_)
print("Bias:", clf.intercept_)

# Plot decision boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define custom color map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['red', 'green'])

# Plot decision region
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=100)
plt.title("Perceptron Decision Region for OR Gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.grid(True)
plt.show()