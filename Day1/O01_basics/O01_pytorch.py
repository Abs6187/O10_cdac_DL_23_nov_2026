'''
nvidia-smi : Check the status of NVIDIA GPU and its memory usage.

pytorch installation steps:
1. Check NVIDIA GPU status: nvidia-smi
2. Install PyTorch: pip install torch torchvision torchaudio
3. Verify installation: python -c "import torch; print(torch.__version__)"

4. python -c "import struct; print(struct.calcsize('P') * 8)" : Check the system architecture (32-bit or 64-bit).
'''

'''
what is PyTorch? 
- PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab. 
- It provides a flexible and efficient platform for building and training neural networks, supporting both CPU and GPU computations.
- It is widely used in both academia and industry for research and production purposes due to its dynamic computation graph and ease of use.
'''

'''
what are tensors in PyTorch?
- Tensors are the fundamental data structures in PyTorch, similar to arrays or matrices in other numerical computing libraries.
- They can be used to represent inputs, outputs, and parameters of neural networks.
- PyTorch provides a wide range of tensor operations, including mathematical, logical, and indexing operations, which can be performed on both CPU and GPU.
- Tensors can also be easily converted between CPU and GPU, enabling efficient computation on different hardware.
'''

'''
Hello, this is a basic PyTorch script.
Each model has 1 input, and learns to extract formula for converting celsius to fahrenheit.
z = w.x + b
# where w is the weight, x is the input, and b is the bias.
F = 1.8 * C + 32
'''

# --- 1. Imports ---
# torch: Core library providing multi-dimensional tensors and automatic differentiation (autograd).
# torch.nn: Neural network module providing building blocks like layers (nn.Linear) and loss functions (nn.MSELoss).
import torch
import torch.nn as nn

# --- 2. Device Configuration ---
# Select NVIDIA CUDA GPU if available; otherwise fall back to CPU.
# Both the model and all data tensors must reside on the same device to avoid runtime errors.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Print the installed PyTorch version
print(torch.__version__)

# --- 3. Training Data (Tensors) ---
# Each scalar value is wrapped in [...] to create a 2D tensor of shape (9, 1) = (num_samples, num_features),
# which is required by nn.Linear (expects a batch dimension first, followed by feature columns).
# dtype=torch.float32 uses standard 32-bit floating-point precision for weights and gradients.
# device=device places the tensors directly onto the selected CPU or GPU.
celsius = torch.tensor(
    [[-20.0], [-10.0], [0.0], [10.0], [20.0], [25.0], [30.0], [37.0], [100.0]],
    dtype=torch.float32,
    device=device
)

# Corresponding target Fahrenheit values, also with shape (9, 1) and dtype float32
fahrenheit = torch.tensor(
    [[-4.0], [14.0], [32.0], [50.0], [68.0], [77.0], [86.0], [98.6], [212.0]],
    dtype=torch.float32,
    device=device
)

# --- 4. Model Definition ---
# Single linear neuron: computes y = w * x + b (F = w * C + b).
# Initializes 1 learnable weight (w) and 1 learnable bias (b) with random values, then moves them to device.
model = nn.Linear(in_features=1, out_features=1)
model.to(device)

# --- 5. Loss Function & Optimizer ---
# MSELoss (Mean Squared Error) measures the average squared difference between predictions and targets.
# It is smooth/differentiable and penalizes larger errors more heavily.
loss_fn = nn.MSELoss()

# SGD (Stochastic Gradient Descent) updates each parameter in model.parameters() via: param = param - lr * grad.
# lr=0.0001 is kept small because input values reach 100.0, making MSE gradients large;
# a larger learning rate would overshoot the minimum and diverge (producing NaN).
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# --- 6. Training Loop (5 Core PyTorch Steps) ---
# Each epoch is one full pass over all 9 samples; many iterations are needed because
# the small learning rate causes the bias (b) to converge gradually toward 32.0.
for epoch in range(1, 50000):
    # Step 1: Clear gradients from the previous iteration (PyTorch accumulates gradients by default)
    optimizer.zero_grad()

    # Step 2: Forward pass — compute predicted Fahrenheit values by passing inputs through the model
    predictions = model(celsius)

    # Step 3: Compute scalar MSE loss comparing predictions to actual Fahrenheit targets
    loss = loss_fn(predictions, fahrenheit)

    # Step 4: Backpropagation — apply the chain rule via autograd to compute d(loss)/dw and d(loss)/db
    loss.backward()

    # Step 5: Update model weight and bias in the opposite direction of their gradients
    optimizer.step()

    # Log progress every 1,000 epochs; .item() extracts a plain Python float from the 0D loss tensor
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# --- 7. Inspect Learned Parameters ---
# .item() converts single-element tensors into Python floats.
# After training, learned_w should approach 1.8000 and learned_b should approach 32.0000.
learned_w = model.weight.item()
learned_b = model.bias.item()
print(f"\nLearned equation: F = {learned_w:.4f} * C + {learned_b:.4f}")

# --- 8. Evaluation / Inference ---
# Switch the model to evaluation mode (good practice; disables Dropout/BatchNorm training behavior in larger models)
model.eval()

# torch.no_grad() disables gradient tracking during inference, saving memory and speeding up computation.
with torch.no_grad():
    # Shape (1, 1) = (batch_size=1, in_features=1); 40°C should predict ~104.00°F
    test_celsius = torch.tensor([[40.0]], dtype=torch.float32, device=device)
    test_fahrenheit = model(test_celsius)
    print(f"Predicted Fahrenheit for 40°C: {test_fahrenheit.item():.2f}")
