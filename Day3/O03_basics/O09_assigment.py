import numpy as np

'''
T20 Cricket Death-Overs Boundary Predictor
The Problem: Death-Overs Boundary Danger (Overs 16–20)
You are building a single-neuron scorer for the final 5 overs of a T20 match. 
Each row represents a ball/over situation with two normalized features (0.0 to 1.0):

Feature 1 (x1): dot_ball_pressure (0.0 = boundaries flowing freely, 1.0 = heavy dot-ball pressure / bowler on top)

Feature 2 (x2): batter_aggression (0.0 = cautious / rotating strike, 1.0 = power-hitter looking to swing big)

Your TODOs
1. Interpret the weights and bias first: Before running any activation function, compute z by hand or in Python. Why is w_death[0] negative (-1.2) while w_death[1] is positive (+2.5)? What does b_death = -0.4 do to the baseline?

2. Pick an activation for a "Boundary Probability" (0 to 1): Suppose a broadcaster wants a smooth 0 to 1 probability that the next delivery goes for a boundary (4 or 6). Which activation function (sigmoid, tanh, or relu) fits best, and why?

3. Pick an activation for a "Momentum Swing" (-1 to +1): Suppose a match tracker wants a signed score where negative values mean bowler momentum, near 0 means balanced, and positive values mean batter momentum. Which activation function should you choose?

4. Test what happens with ReLU: Run single_layer_forward with ReLU (np.maximum(0.0, z)). Look at any 0.0 outputs—what information is lost when z is negative, and when might ReLU still be useful?

5. Change one scenario and re-run: Modify w_death or X_death for a scenario where a set finisher (0.95 aggression) faces a yorker specialist (0.85 pressure), and compare the before/after outputs.
'''
def single_layer_forward(X, w, b, activation_fn):
    z = X @ w + b
    a = activation_fn(z)
    return z, a

# Rows: [dot_ball_pressure, batter_aggression]
X_death = np.array([
    [0.20, 0.30],  # Low pressure, cautious batter
    [0.80, 0.85],  # High pressure, aggressive batter
    [0.75, 0.25],  # High pressure, cautious batter
    [0.30, 0.90],  # Low pressure, aggressive batter
])

w_death = np.array([-1.2, 2.5])
b_death = -0.4