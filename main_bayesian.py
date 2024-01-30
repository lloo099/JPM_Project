import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from skopt import gp_minimize

# Define custom loss function
# Define custom loss# Define custom loss function
def custom_loss(y_true, y_pred, x_input, y_input, u_input):
    df_dx = tf.gradients(y_pred, x_input)[0]
    df_dy = tf.gradients(y_pred, y_input)[0]
    d2f_dy2 = tf.gradients(df_dy, y_input)[0]

    penalty_x_monotonicity = tf.nn.relu(-df_dx)
    penalty_y_monotonicity = tf.nn.relu(df_dy) if tf.math.floormod(tf.cast(u_input, tf.int32), 2) == 0 else 0.0
    penalty_y_convexity = tf.nn.relu(-d2f_dy2) if tf.math.floormod(tf.cast(u_input, tf.int32), 2) == 0 else 0.0

    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    lambda_x = tf.constant(1.0, dtype=tf.float32)  # Weight for x constraint
    lambda_y = tf.constant(1.0, dtype=tf.float32)  # Weight for y constraint

    loss = mse_loss + lambda_x * tf.reduce_mean(penalty_x_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_convexity)

    return loss
# def custom_loss(y_true, y_pred):
#     df_dx = K.gradients(y_pred, x_input)[0]
#     df_dy = K.gradients(y_pred, y_input)[0]
#     d2f_dy2 = K.gradients(df_dy, y_input)[0]

#     penalty_x_monotonicity = tf.nn.relu(-df_dx)
#     penalty_y_monotonicity = tf.nn.relu(df_dy) if tf.math.floormod(tf.cast(u_input, tf.int32), 2) == 0 else 0.0
#     penalty_y_convexity = tf.nn.relu(-d2f_dy2) if tf.math.floormod(tf.cast(u_input, tf.int32), 2) == 0 else 0.0

#     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

#     lambda_x = tf.constant(1.0, dtype=tf.float32)  # Weight for x constraint
#     lambda_y = tf.constant(1.0, dtype=tf.float32)  # Weight for y constraint

#     loss = mse_loss + lambda_x * tf.reduce_mean(penalty_x_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_monotonicity) + lambda_y * tf.reduce_mean(penalty_y_convexity)

#     return loss

# Define your MLP architecture
input_dim = 4  # Corresponding to x, y, u, v
hidden_dim = 64
output_dim = 1

# Generate synthetic training data
num_samples = 10000
x_data = np.random.uniform(-2, 2, size=(num_samples, 1))
y_data = np.random.uniform(0, 1, size=(num_samples, 1))
u_data = np.random.randint(1, 100, size=(num_samples, 1))
v_data = np.random.uniform(-1, 1, size=(num_samples, 1))
f_data = np.random.uniform(-1, 1, size=(num_samples, 1))

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


# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, model.input[0], model.input[1], model.input[2]))

# Train the model
model.fit([x_data, y_data, u_data, v_data], f_data, epochs=10, batch_size=128)

# Define the objective function for optimization
def objective_function(params):
    x, y, u, v = params
    # Ensure u is positive and divisible by 5 but not by 3
    u = max(5, u + (5 - u % 5) % 5)
    # Ensure v is complex
    v = np.complex(v)
    
    # Reshape the inputs for prediction
    x = np.array([[x]])
    y = np.array([[y]])
    u = np.array([[u]])
    v = np.array([[v]])
    
    # Predict using the trained neural network
    prediction = model.predict([x, y, u, v])
    print(prediction)
    
    return -prediction[0, 0]  # Extract the scalar value from the prediction

# Define the bounds for each variable
bounds = [(-2.0, 2.0), (0.0, 1.0), (1, 100), (-1.0, 1.0)]
results = lambda params: objective_function(params)
# Bayesian optimization using a lambda function
res = gp_minimize(lambda params: objective_function(params), bounds, n_calls=10, random_state=1234)

# Get the optimized values
optimal_values = res.x
estimated_maximum = -res.fun  # Negate because gp_minimize minimizes, we want to maximize

# Use the optimal values for further exploration or validation
print("Optimal Values:", optimal_values)
print("Estimated Maximum:", estimated_maximum)

import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_gaussian_process

# Visualize the results of Bayesian optimization
print(f"Minimum Location: x = {res.x[0]}")
print(f"Minimum Function Value: {res.fun}")
# Define the noise level
noise_level = 0.1

# Define the objective function without noise
def f_wo_noise(params):
    x, y, u, v = params
    u = max(5, u + (5 - u % 5) % 5)
    v = np.complex(v)
    
    # Reshape the inputs for prediction
    x = np.array([[x]])
    y = np.array([[y]])
    u = np.array([[u]])
    v = np.array([[v]])
    
    # Predict using the trained neural network
    prediction = model.predict([x, y, u, v])
        # Plot the true function
    plt.contourf(x[:, :, 0, 0], y[:, :, 0, 0], prediction, cmap="viridis")
    plt.colorbar()
    
    return -prediction[0, 0]  # Extract the scalar value from the prediction

# Visualize the results of Bayesian optimization
print(f"Minimum Location: x = {res.x[0]}")
print(f"Minimum Function Value: {res.fun}")

# Convergence plot
plot_convergence(res)
plt.savefig("convergence_plot.png")
plt.savefig("convergence_plot.pdf",dpi=1200)
plt.show()
# Visualize the Gaussian Process and Acquisition Function for each iteration
plt.rcParams["figure.figsize"] = (8, 12)

# for n_iter in range(len(res.models)):
#     # Plot true function.
#     plt.subplot(4, 2, 2 * n_iter + 1)

#     if n_iter == 0:
#         show_legend = True
#     else:
#         show_legend = False






#     # Plot EI(x)
#     print(res)

#     plt.subplot(4, 2, 2 * n_iter + 2)
#     ax = plot_gaussian_process(res, n_calls=n_iter,
#                                objective=f_wo_noise,
#                                show_legend=show_legend, show_title=False,
#                                show_mu=False, show_acq_func=True,
#                                show_observations=False,
#                                show_next_point=True)
#     ax.set_ylabel("")
#     ax.set_xlabel("")

# plt.tight_layout()
# plt.savefig("gp_and_acq_functions.png")
# plt.show()
import matplotlib.pyplot as plt

# Get the best GP model from the optimization results
best_gp_model = res.models[-1]

# Create a grid of points for x, y, u, v
grid_size = 100
xx, yy, uu, vv = np.meshgrid(np.linspace(-2, 2, grid_size),
                             np.linspace(0, 1, grid_size),
                             np.arange(1, 101, 1),
                             np.linspace(-1, 1, grid_size), indexing='ij')

params_grid = np.c_[xx.ravel(), yy.ravel(), uu.ravel(), vv.ravel()]

# Predict using the best GP model
predictions, _ = best_gp_model.predict(params_grid, return_std=True)

# Reshape predictions to match the shape of the grid
predictions = predictions.reshape(xx.shape)

# Plotting
fig, axes = plt.subplots(4, 4, figsize=(15, 15), sharex=True, sharey=True)

for i in range(4):
    for j in range(4):
        if i == j:
            # Diagonal: plot the marginal distribution
            axes[i, i].hist(res.space.transform(params_grid)[:, i], bins=30, color='skyblue', density=True)
            axes[i, i].set_title(f"{res.space.dimension_names[i]} Marginal Distribution")
        else:
            # Off-diagonal: plot the GP approximation
            scatter = axes[i, j].scatter(res.space.transform(res.x_iters)[:, j],
                                         res.space.transform(res.x_iters)[:, i],
                                         c=res.func_vals, cmap="viridis", marker='o', s=100, edgecolors='k')
            axes[i, j].contourf(xx[:, :, 0, 0], yy[:, :, 0, 0], predictions[0, :, :, 0], cmap="viridis", alpha=0.5)
            axes[i, j].set_xlabel(res.space.dimension_names[j])
            axes[i, j].set_ylabel(res.space.dimension_names[i])
            fig.colorbar(scatter, ax=axes[i, j], label='Function Value')

plt.tight_layout()
plt.savefig("gp_pair_plot.png")
plt.savefig("gp_pair_plot.pdf",dpi=1200)
plt.show()
