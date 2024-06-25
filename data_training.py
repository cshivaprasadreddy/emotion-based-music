import os
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables
is_init = False
label = []
dictionary = {}
c = 0

# Function to check if all elements in the array are numeric
def is_numeric_array(arr):
    return np.issubdtype(arr.dtype, np.number)

# Load data from .npy files
for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):
        data = np.load(i)
        
        if not is_numeric_array(data):
            print(f"Error: Non-numeric data in file {i}")
            continue
        
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Reshape to 2D if necessary

        if not is_init:
            is_init = True
            X = data
            expected_dim = X.shape[1]
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            if data.shape[1] != expected_dim:
                print(f"Error: Dimension mismatch in file {i}. Expected {expected_dim}, got {data.shape[1]}")
                continue
            X = np.concatenate((X, data))
            size = data.shape[0]
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Convert labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode labels
y = to_categorical(y)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Define the model
input_layer = Input(shape=(X.shape[1],))
dense_layer1 = Dense(512, activation="relu")(input_layer)
dense_layer2 = Dense(256, activation="relu")(dense_layer1)
output_layer = Dense(y.shape[1], activation="softmax")(dense_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
