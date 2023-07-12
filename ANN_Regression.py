import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score


# Load the combined CSV and NPZ data
combined_df = pd.read_csv("combined_data.csv")
combined_data = np.load("combined_data.npz")['data']

# Separate the features (input) and labels (output)
features = combined_data[:, 2:]  # Remove the first two elements from each row
labels = combined_df[['HII_EFF_FACTOR', 'ION_Tvir_MIN', 'R_BUBBLE_MAX']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(20,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3)  # 3 output variables
])


# # Reshape the input data to match the expected shape for 1D CNN
# X_train = np.expand_dims(X_train, axis=2)
# X_test = np.expand_dims(X_test, axis=2)

# # Define the neural network architecture (1D CNN)
# model = keras.Sequential([
#     keras.layers.Conv1D(64, 3, activation='relu', input_shape=(20, 1)),
#     keras.layers.MaxPooling1D(2),
#     keras.layers.Conv1D(32, 3, activation='relu'),
#     keras.layers.MaxPooling1D(2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(3)  # 3 output variables
# ])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the model
model.save('my_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
loss = model.evaluate(X_test, y_test, verbose=0)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Loss:", loss)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)