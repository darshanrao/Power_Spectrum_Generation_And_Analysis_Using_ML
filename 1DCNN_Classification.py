import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the binned data from the CSV file
data = pd.read_csv('binned_data.csv')

input_data = np.load("combined_data.npz")['data']

# Separate the features (input) and labels (output)
features = input_data[:, 2:]  # Remove the first two elements from each row

# Split the data into train and validation sets
X_train, X_test, labels_train, labels_test = train_test_split(features, data, test_size=0.2, random_state=42)
labels_train = [np.array(labels_train['HII_EFF_FACTOR_bin']-1), np.array(labels_train['ION_Tvir_MIN_bin']-1), np.array(labels_train['R_BUBBLE_MAX_bin']-1)]
labels_test = [np.array(labels_test['HII_EFF_FACTOR_bin']-1), np.array(labels_test['ION_Tvir_MIN_bin']-1), np.array(labels_test['R_BUBBLE_MAX_bin']-1)]

# Reshape the input data for CNN (assuming 1D sequence data)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Scale the input data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Define the number of classes for each variable
num_classes_parameter1 = len(data['HII_EFF_FACTOR_bin'].unique())
num_classes_parameter2 = len(data['ION_Tvir_MIN_bin'].unique())
num_classes_parameter3 = len(data['R_BUBBLE_MAX_bin'].unique())

# Build the CNN model
visible = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Conv1D(32, kernel_size=3, activation='relu')(visible)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(64, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output1 = Dense(num_classes_parameter1, activation='softmax', name='parameter1')(x)
output2 = Dense(num_classes_parameter2, activation='softmax', name='parameter2')(x)
output3 = Dense(num_classes_parameter3, activation='softmax', name='parameter3')(x)

model = Model(inputs=visible, outputs=[output1, output2, output3])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['sparse_categorical_accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, labels_train, epochs=1000, validation_data=(X_test_scaled, labels_test), verbose=2, callbacks=[early_stopping])

# Save the trained model
model.save('classification_model.h5')
