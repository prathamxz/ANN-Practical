# ✅ How to Implement
# 	1.	Install required packages (if not already installed): pip install tensorflow scikit-learn matplotlib
# 	2.	Save the code in a Python file like logistic_tf.py.
# 	3.	Run it: python logistic_tf.py



import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Generate synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Step 2: Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build logistic regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X.shape[1],))
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model and save the training history
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Step 7: Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# Step 8: Show results
print("\nTest Accuracy:", accuracy_score(y_test, y_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))

# Step 9: Plot the training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()