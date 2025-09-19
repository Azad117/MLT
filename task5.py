import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.utils import to_categorical

# Load dataset
digits = load_digits()
df = pd.DataFrame(digits.data)
df["label"] = digits.target

# Save to CSV
df.to_csv("image_data.csv", index=False)

# Read CSV
data = pd.read_csv("image_data.csv")
x = data.drop("label", axis=1).values / 16.0  # Normalize
y = to_categorical(data["label"].values)      # One-hot encode

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Build model
model = Sequential([
    Input(shape=(64,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(x_train, y_train, epochs=20, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy: {accuracy:.4f}")

# Make a prediction
pred = model.predict(x_test[:1])
print("Predicted label:", np.argmax(pred), "True Label:", np.argmax(y_test[0]))




'''
Output:

Accuracy: 0.9667
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step
Predicted label: 2 True Label: 2

'''
