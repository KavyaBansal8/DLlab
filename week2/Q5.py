import torch

x = torch.tensor(1.0, requires_grad=True)

y = 8 * x**4 + 3 * x**3 + 7 * x**2 + 6 * x + 3

y.backward()

# Print the gradient
print(f"Gradient dy/dx (using PyTorch): {x.grad.item()}")

# Analytical gradient for verification
analytical_gradient = 32 * x.item()**3 + 9 * x.item()**2 + 14 * x.item() + 6
print(f"Analytical gradient dy/dx: {analytical_gradient}")

