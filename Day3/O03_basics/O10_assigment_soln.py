import numpy as np

'''
T20 Cricket Death-Overs Boundary Predictor
The Problem: Death-Overs Boundary Danger (Overs 16–20)
You are building a single-neuron scorer for the final 5 overs of a T20 match. 
Each row represents a ball/over situation with two normalized features (0.0 to 1.0):

Feature 1 (x1): dot_ball_pressure (0.0 = boundaries flowing freely, 1.0 = heavy dot-ball pressure / bowler on top)

Feature 2 (x2): batter_aggression (0.0 = cautious / rotating strike, 1.0 = power-hitter looking to swing big)
'''
def single_layer_forward(X, w, b, activation_fn):
    z = X @ w + b
    a = activation_fn(z)
    return z, a

# Activation functions
linear = lambda z: z
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
tanh_fn = np.tanh
relu = lambda z: np.maximum(0.0, z)

# Rows: [dot_ball_pressure, batter_aggression]
X_death = np.array([
    [0.20, 0.30],  # Low pressure, cautious batter
    [0.80, 0.85],  # High pressure, aggressive batter
    [0.75, 0.25],  # High pressure, cautious batter
    [0.30, 0.90],  # Low pressure, aggressive batter
])

'''
why is w_death[0] negative and w_death[1] positive, and what does b_death = -0.4 do?
-> -1.2 means that higher dot-ball pressure reduces the score (less likely to hit a boundary)
-> 2.5 means that higher batter aggression increases the score (more likely to hit a boundary)
-> b_death = -0.4 sets the baseline score lower, making boundaries less likely by default
'''
w_death = np.array([-1.2, 2.5])
b_death = -0.4

print("Linear Activation (z = Xw + b)")
z_linear_pre, z_linear_post = single_layer_forward(X_death, w_death, b_death, linear)
print("TODO 1 - Pre-activations z:", np.round(z_linear_pre, 4))
print("TODO 1 - Post-activations a:", np.round(z_linear_post, 4))
print("_" * 60)

# TODO 2 - Boundary Probability with Sigmoid
print("Sigmoid Activation (Boundary Probability)")
z_sigmoid_pre, z_sigmoid_post = single_layer_forward(X_death, w_death, b_death, sigmoid)
print("TODO 2 - Pre-activations z (sigmoid):", np.round(z_sigmoid_pre, 4))
print("TODO 2 - Post-activations a (sigmoid):", np.round(z_sigmoid_post, 4))
print("_" * 60)

# TODO 3 - Momentum Swing with Tanh
print("Tanh Activation (Momentum Swing)")
z_tanh_pre, z_tanh_post = single_layer_forward(X_death, w_death, b_death, tanh_fn)
print("TODO 3 - Pre-activations z (tanh):", np.round(z_tanh_pre, 4))
print("TODO 3 - Post-activations a (tanh):", np.round(z_tanh_post, 4))
print("_" * 60)

# TODO 4 - Test with ReLU
print("ReLU Activation (Test with ReLU)")
z_relu_pre, z_relu_post = single_layer_forward(X_death, w_death, b_death, relu)
print("TODO 4 - Pre-activations z (ReLU):", np.round(z_relu_pre, 4))
print("TODO 4 - Post-activations a (ReLU):", np.round(z_relu_post, 4))
print("_" * 60)


print("\nTODO 5 - Adding a new scenario (5th row):")
X_death_v2 = np.vstack([
    X_death,
    [0.85, 0.95],  # High pressure (0.85), high aggression (0.95)
])

z_v2, a_sig_v2 = single_layer_forward(X_death_v2, w_death, b_death, sigmoid)
_, a_tanh_v2 = single_layer_forward(X_death_v2, w_death, b_death, tanh_fn)
_, a_relu_v2 = single_layer_forward(X_death_v2, w_death, b_death, relu)
print("\nTODO 5 - Updated scenarios (5th row is finisher vs. yorker specialist):")
print("  z (linear):       ", np.round(z_v2, 4))
print("  a (sigmoid):      ", np.round(a_sig_v2, 4))
print("  a (tanh):         ", np.round(a_tanh_v2, 4))
print("  a (ReLU):         ", np.round(a_relu_v2, 4))
print("_" * 60)