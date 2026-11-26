import numpy as np

X_train = np.array([
    [0.2, 0.2],  # Pair 1: No
    [0.5, 0.4],  # Pair 2: No
    [0.8, 0.7],  # Pair 3: Yes
])
y_train = np.array([0, 0, 1])  # 0 = NO, 1 = YES

w = np.array([2.0, 1.0])
raw_train = X_train @ w

print("STEP 1: Raw Signal Before Bias (X_train @ w)")
print(f"Weights w = {w} (cloud={w[0]}, wind={w[1]})\n")
for i, (x_row, r_val, y_val) in enumerate(zip(X_train, raw_train, y_train), start=1):
    want = "NO  (needs z < 0)" if y_val == 0 else "YES (needs z > 0)"
    expr = f"({w[0]:.1f})({x_row[0]:.1f}) + ({w[1]:.1f})({x_row[1]:.1f})"
    print(f"  Pair {i} {x_row}: {expr} = {r_val:.2f}  -> Want: {want}")

max_no_raw = raw_train[y_train == 0].max()
min_yes_raw = raw_train[y_train == 1].min()
tipping_point = (max_no_raw + min_yes_raw) / 2.0
b = -tipping_point

print("\nSTEP 2: Deciding the Bias (b)")
print(f"  Raw sums: {[round(float(v), 2) for v in raw_train]}")
print(f"  Highest NO raw sum  = {max_no_raw:.2f} (must stay below 0 after adding b)")
print(f"  Lowest  YES raw sum = {min_yes_raw:.2f} (must rise above 0 after adding b)")
print(f"  Midpoint tipping point = ({max_no_raw:.2f} + {min_yes_raw:.2f}) / 2 = {tipping_point:.2f}")
print(f"  Set bias b = -{tipping_point:.2f} = {b:.2f} (shifts {tipping_point:.2f} down to 0.00)\n")

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
z_train = raw_train + b
a_train = sigmoid(z_train)

print("STEP 3: Training Verification (Pre-Activations z & Sigmoid Outputs)")
train_correct = 0
for i in range(len(X_train)):
    pred = 1 if a_train[i] >= 0.5 else 0
    decision = "YES (1)" if pred == 1 else "NO  (0)"
    actual = "YES (1)" if y_train[i] == 1 else "NO  (0)"
    match = "PASS" if pred == y_train[i] else "FAIL"
    if pred == y_train[i]:
        train_correct += 1
    print(
        f"  Pair {i+1} {X_train[i]} -> raw={raw_train[i]:.2f}, "
        f"z={z_train[i]:+.2f} ({raw_train[i]:.2f} + ({b:.2f})), "
        f"sigmoid={a_train[i]:.2f} -> pred={decision}, actual={actual} [{match}]"
    )
print(f"  Training accuracy: {train_correct}/{len(X_train)}")

X_new = np.array([
    [0.1, 0.3],  # New 1: Clear sky, light wind -> expect NO (0)
    [0.6, 0.5],  # New 2: Moderate clouds + breeze -> expect NO (0)
    [0.7, 0.6],  # New 3: Fairly cloudy + breezy -> expect YES (1)
    [0.9, 0.8],  # New 4: Heavy clouds + strong wind -> expect YES (1)
])
y_new = np.array([0, 0, 1, 1])

raw_new = X_new @ w
z_new = raw_new + b
a_new = sigmoid(z_new)

print("\nSTEP 4: New Unseen Data Verification")
new_correct = 0
for i, (x_row, r_val, z_val, a_val, y_val) in enumerate(
    zip(X_new, raw_new, z_new, a_new, y_new), start=1
):
    pred = 1 if a_val >= 0.5 else 0
    decision = "YES (1)" if pred == 1 else "NO  (0)"
    expected = "YES (1)" if y_val == 1 else "NO  (0)"
    match = "PASS" if pred == y_val else "FAIL"
    if pred == y_val:
        new_correct += 1
    print(
        f"  New {i} {x_row} -> raw={r_val:.2f}, "
        f"z={z_val:+.2f} ({r_val:.2f} + ({b:.2f})), "
        f"sigmoid={a_val:.2f} -> pred={decision}, expected={expected} [{match}]"
    )
print(f"  New data accuracy: {new_correct}/{len(X_new)}")