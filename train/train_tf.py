import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load and preprocess the data
data = pd.read_csv('./data/train.csv')
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.dayofweek
data['year'] = data['date'].dt.year

# Define features and target
features = ['store', 'item', 'month', 'day', 'year']
target = 'sales'

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=2018)

train_x = train_data[features]
train_y = train_data[target]
test_x = test_data[features]
test_y = test_data[target]

# Convert to TensorFlow dataset in proper format
train_ds = tf.data.Dataset.from_tensor_slices((train_x.values, train_y)).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((test_x.values, test_y)).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(features),)),  # Input shape matches the number of features
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mean_absolute_percentage_error']
)

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=500,
    callbacks=[early_stopping]
)
print("Model training complete.")

A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A

