import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras import backend as K

# Define your MLP architecture
input_dim = 4  # Corresponding to x, y, u, v
hidden_dim = 64
output_dim = 10
size=10
# Define input layers using tf.keras.Input
x_input = tf.keras.Input(shape=(size,), name='x_input')
y_input = tf.keras.Input(shape=(size,), name='y_input')
u_input = tf.keras.Input(shape=(size,), name='u_input')
v_input = tf.keras.Input(shape=(size,), name='v_input')

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

    # Include observational noise in the loss
    noise_variance = 0.1
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred) / (2 * noise_variance) + 0.5 * tf.math.log(noise_variance))

    lambda_x = tf.constant(1.0, dtype=tf.float32)  # Weight for x constraint
    lambda_y = tf.constant(1.0, dtype=tf.float32)  # Weight for y constraint

    loss = mse_loss + lambda_x * tf.reduce_mean(penalty_x_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_convexity)

    return loss

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=custom_loss)

# Generate synthetic training data with noise
num_samples = 100
x_data = np.random.uniform(0, 1, size=(num_samples, size))
y_data = np.random.uniform(0, 1, size=(num_samples, size))
u_data = np.random.randint(1, 100, size=(num_samples, size))
v_data = np.random.uniform(0, 1, size=(num_samples, size))
f_data = np.random.uniform(0, 1, size=(num_samples, 10))

# Add observational noise


def dynamic_normal_distribution(num_samples):
    """
    Generate random numbers from a normal distribution with dynamically changing standard deviation.

    Parameters:
    - num_samples: Number of samples to generate.

    Returns:
    - NumPy array of generated samples.
    """
    samples = []
    current_mean = 0.0
    initial_std_dev = 1.0  # Initial standard deviation

    for _ in range(num_samples):
        # Generate a random number from the normal distribution with the current mean and standard deviation
        sample = np.random.normal(current_mean, initial_std_dev)
        samples.append(sample)

        # Update the standard deviation dynamically
        initial_std_dev = update_std_dev(initial_std_dev)

    return np.array(samples)

def update_std_dev(prior_std_dev):
    """
    Update the standard deviation dynamically based on the specified formula.

    Parameters:
    - prior_std_dev: Prior standard deviation.

    Returns:
    - Updated standard deviation.
    """
    updated_std_dev = 0.1 + 0.8 * prior_std_dev + 0.1 * prior_std_dev
    return updated_std_dev



# Generate samples from the dynamically changing normal distribution
noise = dynamic_normal_distribution(10)
# Ensure both arrays have the same shape
# if f_data.shape != noise.shape:
#     min_samples = min(f_data.shape[0], noise.shape[0])
#     f_data = f_data[:min_samples]
#     noise = noise[:min_samples]
# Print the generated samples
# print(generated_samples)

# noise = 0.1 * np.random.normal(size=(num_samples, 1))
observed_data = f_data + noise

# Train the model
history=model.fit([x_data, y_data, u_data, v_data], observed_data, epochs=100, batch_size=10)




import matplotlib.pyplot as plt


plt.plot(history.history['loss'],label='Number of Samples = 100')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best',fontsize=12)
# plt.savefig('train_loss_withoutnoise.png')
plt.savefig('train_loss_withnoise_100epo.png')


# Assuming x_data, y_data, u_data, v_data, and observed_data are available
predictions = model.predict([x_data, y_data, u_data, v_data])
print(predictions.shape)
print(observed_data.shape)
plt.scatter(observed_data, predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.savefig('scatter_plot_10.png')

plt.show()
