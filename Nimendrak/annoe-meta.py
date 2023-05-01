import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# Load the data
data = pd.read_csv("../data/mobilenetv2.csv")

# Map optimizer names to integers
optimizer_mapping = {optimizer: idx for idx, optimizer in enumerate(data['optimizer'].unique())}
data['optimizer'] = data['optimizer'].map(optimizer_mapping)

# Standardize the input features
scaler = StandardScaler()
data[['cpu', 'memory', 'dataset']] = scaler.fit_transform(data[['cpu', 'memory', 'dataset']])

# Normalize the output features
output_scaler = MinMaxScaler()
data[['learning_rate', 'momentum', 'time', 'epochs', 'loss', 'accuracy']] = output_scaler.fit_transform(data[['learning_rate', 'momentum', 'time', 'epochs', 'loss', 'accuracy']])

# Prepare the input and output features
input_features = ['cpu', 'memory', 'dataset']
output_features = ['optimizer', 'learning_rate', 'momentum', 'time', 'epochs', 'loss', 'accuracy']

X = data[input_features]
y = data[output_features]

# One-hot encode the optimizer labels
y_optimizer_one_hot = to_categorical(y['optimizer'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate the optimizer one-hot labels for training and testing
y_train_optimizer_one_hot = y_optimizer_one_hot[y_train.index]
y_test_optimizer_one_hot = y_optimizer_one_hot[y_test.index]

# Define the neural network model
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer = Dense(64, activation='relu')(input_layer)
hidden_layer = Dense(32, activation='relu')(hidden_layer)

# Multi-task outputs
optimizer_output = Dense(len(optimizer_mapping), activation='softmax', name='optimizer_output')(hidden_layer)
other_outputs = Dense(len(output_features) - 1, activation='linear', name='other_outputs')(hidden_layer)

# Create the model
model = Model(inputs=input_layer, outputs=[optimizer_output, other_outputs])

# Compile the model
model.compile(optimizer='adam',
              loss={'optimizer_output': 'categorical_crossentropy', 'other_outputs': 'mse'},
              metrics={'optimizer_output': 'accuracy', 'other_outputs': 'mse'})

# Train the model
model.fit(X_train, [y_train_optimizer_one_hot, y_train.drop(columns='optimizer')],
          epochs=100, batch_size=32, validation_split=0.2)

# Make predictions
y_pred_proba_optimizer, y_pred_other = model.predict(X_test)

# Get optimizer predictions
y_pred_optimizer = np.argmax(y_pred_proba_optimizer, axis=1)

# Calculate the accuracy
acc = accuracy_score(y_test['optimizer'], y_pred_optimizer)
print("Accuracy: ", acc)

# Function to predict the optimizer and other features given cpu, memory, and dataset size
def predict_all(cpu, memory, dataset_size):
    input_data = np.array([[cpu, memory, dataset_size]])
    input_data_scaled = scaler.transform(input_data)
    prediction_optimizer_proba, prediction_other = model.predict(input_data_scaled)

    # Get the predicted optimizer
    optimizer = np.argmax(prediction_optimizer_proba, axis=1)[0]
    optimizer_name = {value: key for key, value in optimizer_mapping.items()}[optimizer]

    # Get the predicted other features
    predicted_other = output_scaler.inverse_transform(prediction_other)

    learning_rate, momentum, time, epochs, loss, accuracy = predicted_other[0]

    return optimizer, optimizer_name, learning_rate, momentum, time, epochs, loss, accuracy

# Example prediction
optimizer, optimizer_name, learning_rate, momentum, time, epochs, loss, accuracy = predict_all(cpu=60, memory=40, dataset_size=300)

print("Predicted optimizer:", optimizer_name)
print("Predicted learning rate:", learning_rate)
print("Predicted momentum:", momentum)
print("Predicted time:", time)
print("Predicted epochs:", epochs)
print("Predicted loss:", loss)
print("Predicted accuracy:", accuracy)