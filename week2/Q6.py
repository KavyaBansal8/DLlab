import torch

# Define inputs as tensors with gradient tracking
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

# Forward pass
a = 2 * x
b = torch.sin(y)
c = a / b
d = c * z
e = torch.log(d + 1)
f = torch.tanh(e)

# Backward pass (compute gradients)
f.backward()

# Print the gradients
print(f"Gradient of f w.r.t y (PyTorch): {y.grad.item()}")

# Analytical gradient for verification
print(f"Analytical Gradient of f w.r.t y: {-0.134}")
