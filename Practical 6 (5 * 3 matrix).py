# ▶ How to Implement
# 	1.	Install scikit-learn (if not already): pip install scikit-learn
# 	2.	Save as: digit_recognizer.py
# 	3.	Run it: python digit_recognizer.py


import numpy as np
from sklearn.neural_network import MLPClassifier

# Each digit is a 5x3 binary matrix
digits = {
    "0": [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    "1": [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1]
    ],
    "2": [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ],
    "3": [
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ],
    "9": [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
}

# Prepare training data
X = []
y = []

for label, matrix in digits.items():
    flat = np.array(matrix).flatten()
    X.append(flat)
    y.append(label)

# Train MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000, random_state=42)
clf.fit(X, y)

# Test function
def test_digit(test_matrix):
    flat = np.array(test_matrix).flatten().reshape(1, -1)
    prediction = clf.predict(flat)
    print("Recognized as:", prediction[0])

# Example test
test_data = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1]
]
print("Test Digit:")
for row in test_data:
    print(row)
test_digit(test_data)