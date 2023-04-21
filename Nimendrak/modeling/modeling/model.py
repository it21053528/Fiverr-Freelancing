import os

# Suppress the warning about using the python implementation of protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model

from plots.histrogram import histrogram

# Load the dataset
data = pd.read_csv('data/generated_dataset.csv')
# data = pd.read_csv('data/mobilenetv2_pre.csv')

# Preprocess the data
X = data[['cpu', 'memory', 'dataset']]
y = data['optimizer']

# Scale the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the optimizer labels
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
num_classes = y_encoded.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.3))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

if not os.path.exists('model.h5'):
  # Compile the model
  model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
  )

  # Train the model
  history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    validation_data=(X_test, y_test)
  )

  # Plot the model
  histrogram(history)
  
  loss, accuracy = model.evaluate(X_test, y_test)
  print("Test loss:", loss)
  print("Test accuracy:", accuracy)

  # Save the trained model
  model.save('model.h5')

  # Save the fitted scaler and encoder objects
  with open('scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)
  with open('encoder.pkl', 'wb') as f:
      pickle.dump(encoder, f)

  model.summary()

# Make predictions
def predict_best_optimizer(model, cpu, memory, dataset):
    input_data = np.array([[cpu, memory, dataset]])
    input_scaled = scaler.transform(input_data)
    probabilities = model.predict(input_scaled)
    best_optimizer_idx = np.argmax(probabilities)
    best_optimizer_onehot = np.zeros((1, probabilities.shape[1]))
    best_optimizer_onehot[0, best_optimizer_idx] = 1
    best_optimizer = encoder.inverse_transform(best_optimizer_onehot)[0][0]
    return best_optimizer

# Load the saved model
loaded_model = load_model('model.h5')
# Example usage
best_optimizer = predict_best_optimizer(loaded_model, cpu=10, memory=20, dataset=200)
print("Best optimizer:", best_optimizer)

if os.path.exists('model.h5'):
  os.remove('model.h5')

if os.path.exists('scaler.pkl'):
  os.remove('scaler.pkl')

if os.path.exists('encoder.pkl'):
  os.remove('encoder.pkl')