import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Define your MLP architecture
input_dim = 4  # Corresponding to x, y, u, v
hidden_dim = 64
output_dim = 1

# Define input layers using tf.keras.Input
x_input = tf.keras.Input(shape=(1,), name='x_input')
y_input = tf.keras.Input(shape=(1,), name='y_input')
u_input = tf.keras.Input(shape=(1,), name='u_input')
v_input = tf.keras.Input(shape=(1,), name='v_input')

# Combine input layers
input_concat = tf.keras.layers.Concatenate()([x_input, y_input, u_input, v_input])
hidden1 = tf.keras.layers.Dense(hidden_dim, activation='relu')(input_concat)
hidden2 = tf.keras.layers.Dense(hidden_dim, activation='relu')(hidden1)
output = tf.keras.layers.Dense(output_dim)(hidden2)

# Define the model
model = tf.keras.Model(inputs=[x_input, y_input, u_input, v_input], outputs=output)

# Define custom loss
def custom_loss(y_true, y_pred):
    df_dx = K.gradients(y_pred, x_input)[0]
    df_dy = K.gradients(y_pred, y_input)[0]
    d2f_dy2 = K.gradients(df_dy, y_input)[0]

    penalty_x_monotonicity = tf.nn.relu(-df_dx)
    penalty_y_monotonicity = tf.nn.relu(df_dy) if tf.math.floormod(tf.cast(u_input, tf.int32), 2) == 0 else 0.0
    penalty_y_convexity = tf.nn.relu(-d2f_dy2) if tf.math.floormod(tf.cast(u_input, tf.int32), 2) == 0 else 0.0

    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    lambda_x = tf.constant(1.0, dtype=tf.float32)  # Weight for x constraint
    lambda_y = tf.constant(1.0, dtype=tf.float32)  # Weight for y constraint

    loss = mse_loss + lambda_x * tf.reduce_mean(penalty_x_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_convexity)

    return loss

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=custom_loss)

# Generate synthetic training data
num_samples0 = 100
x_data0 = np.random.uniform(0, 1, size=(num_samples0, 1))
y_data0 = np.random.uniform(0, 1, size=(num_samples0, 1))
u_data0 = np.random.randint(1, 100, size=(num_samples0, 1))
v_data0 = np.random.uniform(0, 1, size=(num_samples0, 1))
f_data0 = np.random.uniform(0, 1, size=(num_samples0, 1))

# Generate synthetic training data
num_samples = 1000
x_data = np.random.uniform(0, 1, size=(num_samples, 1))
y_data = np.random.uniform(0, 1, size=(num_samples, 1))
u_data = np.random.randint(1, 100, size=(num_samples, 1))
v_data = np.random.uniform(0, 1, size=(num_samples, 1))
f_data = np.random.uniform(0, 1, size=(num_samples, 1))

# Generate synthetic training data
num_samples2 = 10000
x_data2 = np.random.uniform(0, 1, size=(num_samples2, 1))
y_data2 = np.random.uniform(0, 1, size=(num_samples2, 1))
u_data2 = np.random.randint(1, 100, size=(num_samples2, 1))
v_data2 = np.random.uniform(0, 1, size=(num_samples2, 1))
f_data2 = np.random.uniform(0, 1, size=(num_samples2, 1))
# # Train the model
# model.fit([x_data, y_data, u_data, v_data], f_data, epochs=1000, batch_size=128)

# Train the model and store the training history
# history0 = model.fit([x_data0, y_data0, u_data0, v_data0], f_data0, epochs=50, batch_size=100)
# history = model.fit([x_data, y_data, u_data, v_data], f_data, epochs=50, batch_size=100)
history2 = model.fit([x_data2, y_data2, u_data2, v_data2], f_data2, epochs=100, batch_size=100)
# Plot the training loss
# plt.plot(history0.history['loss'],label='Number of Samples = 100')
# plt.plot(history.history['loss'],label='Number of Samples = 1000')
plt.plot(history2.history['loss'],label='Number of Samples = 10000')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best',fontsize=12)
# plt.savefig('train_loss_withoutnoise.png')
plt.savefig('train_loss_withoutnoise_100epo.png')


plt.show()