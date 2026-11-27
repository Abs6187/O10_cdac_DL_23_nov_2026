'''
Deciding whether to carry an umbrella based on weather conditions

x1 : Cloud Cover (0.0 = clear sky, 1.0 = dark overcast)
x2 : Wind Speed (0.0 = calm, 1.0 = very windy)

'''

import numpy as np

X = np.array([
    [0.2, 0.2],  # Pair 1: No
    [0.5, 0.4],  # Pair 2: No
    [0.8, 0.7],  # Pair 3: Yes
])
labels = np.array([0, 0, 1])  # 0 = No umbrella, 1 = Yes umbrella
w = np.array([2.0, 1.0])
raw = X @ w
print("Raw scores:", np.round(raw, 4))
print(f"Weights w = {w} (cloud={w[0]}, wind={w[1]})\n")
print("Labels:", labels)
print("_" * 60)
for i, (x_row,r_val,y_val) in enumerate(zip(X, raw, labels),start=1):
    want = "NO  (needs z < 0)" if y_val == 0 else "YES (needs z > 0)"
    expr = f"({w[0]:.1f})({x_row[0]:.1f}) + ({w[1]:.1f})({x_row[1]:.1f})"
    print(f"  Pair {i} {x_row}: {expr} = {r_val:.2f}  -> Want: {want}")

print("_" * 60)
max_no_raw = raw[labels == 0].max()
min_yes_raw = raw[labels == 1].min()
print(f"Max raw score for 'No' umbrella: {max_no_raw:.2f}")
print(f"Min raw score for 'Yes' umbrella: {min_yes_raw:.2f}")
print("_" * 60)
tipping_point = (max_no_raw + min_yes_raw) / 2.0
b = -tipping_point
print("\nSTEP 2: Deciding the Bias (b)")
print(f"  Raw sums: {[round(float(v), 2) for v in raw]}")
print(f"  Highest NO raw sum  = {max_no_raw:.2f} (must stay below 0 after adding b)")
print(f"  Lowest  YES raw sum = {min_yes_raw:.2f} (must rise above 0 after adding b)")
print(f"  Midpoint tipping point = ({max_no_raw:.2f} + {min_yes_raw:.2f}) / 2 = {tipping_point:.2f}")
print(f"  Set bias b = -{tipping_point:.2f} = {b:.2f} (shifts {tipping_point:.2f} down to 0.00)\n")
print("_" * 60)

# Step 3: Compute pre-activation z and sigmoid output
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
z = raw + b
a = sigmoid(z)
print("_" * 60)

print("STEP 3: Final Pre-Activations (z) and Sigmoid Outputs")
for i in range(len(X)):
    decision = "YES" if a[i] >= 0.5 else "NO"
    print(
        f"  Pair {i+1} {X[i]} -> raw={raw[i]:.2f}, "
        f"z={z[i]:+.2f} ({raw[i]:.2f} + ({b:.2f})), "
        f"sigmoid={a[i]:.2f} -> {decision}"
    )
print("_" * 60)
