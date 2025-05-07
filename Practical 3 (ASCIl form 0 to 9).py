# ▶ How to Implement
# 	1.	Install scikit-learn (if not installed): pip install scikit-learn
# 	2.	Save as: even_odd_perceptron.py
# 	3.	Run: python even_odd_perceptron.py
# 	4.	Expected Output:
# Digit | Prediction | Actual
# ----------------------------
#   0   |     0      |   0
#   1   |     1      |   1
#   2   |     0      |   0
#   3   |     1      |   1
#   4   |     0      |   0
#   5   |     1      |   1
#   6   |     0      |   0
#   7   |     1      |   1
#   8   |     0      |   0
#   9   |     1      |   1


import numpy as np
from sklearn.linear_model import Perceptron

# Convert a digit to its ASCII binary representation (8 bits)
def ascii_binary(digit_char):
    ascii_code = ord(digit_char)
    binary = [int(bit) for bit in f"{ascii_code:08b}"]
    return binary

# Prepare dataset (0–9)
digits = [str(d) for d in range(10)]
X = [ascii_binary(d) for d in digits]
y = [int(d) % 2 for d in digits]  # Label: 0 for even, 1 for odd

# Train Perceptron
model = Perceptron(max_iter=1000, tol=1e-3)
model.fit(X, y)

# Test the model
print("Digit | Prediction | Actual")
print("----------------------------")
for d in digits:
    input_data = np.array(ascii_binary(d)).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    print(f"  {d}   |     {prediction}      |   {int(d)%2}")