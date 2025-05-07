# ▶ How to Implement
# 	1.	Install TensorFlow: pip install tensorflow scikit-learn
# 	2.	Save the above code as logistic_regression_tf.py.
# 	3.	Run the program:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()