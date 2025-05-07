# ▶ How to Implement
# 1.	Save the code in a file named andnot_mcp.py.
# 2.	Run the program in a terminal: python andnot_mcp.py
# 3.	Expected Output:
# x1 x2 | ANDNOT
# --------------
# 0  0  |   1
# 0  1  |   0
# 1  0  |   1
# 1  1  |   0

# This simulates a single McCulloch-Pitts neuron with:
# 	•	Weights: w1 = 1, w2 = -1
# 	•	Threshold: 1
# 	•	Activation function: binary step (0 or 1)


def mcculloch_pitts_andnot(x1, x2):
    # Assign weights (1 for x1, -1 for x2)
    w1 = 1
    w2 = -1
    threshold = 1  # Set threshold

    # Compute weighted sum
    net_input = w1 * x1 + w2 * x2

    # Apply step function (binary threshold)
    output = 1 if net_input >= threshold else 0

    return output

# Test all input combinations
print("x1 x2 | ANDNOT")
print("--------------")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        result = mcculloch_pitts_andnot(x1, x2)
        print(f"{x1}  {x2}  |   {result}")