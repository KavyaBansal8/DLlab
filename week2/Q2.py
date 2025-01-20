import torch

# Inputs
x = torch.tensor(2.0, requires_grad=True)  # Example value for x
w = torch.tensor(3.0, requires_grad=True)  # Example value for w
b = torch.tensor(1.0, requires_grad=True)  # Example value for b

# Computation
u = w * x
v = u + b
a = torch.relu(v)

# Backpropagation
a.backward()

# Gradients
print(f"Gradient da/dw (using PyTorch): {w.grad.item()}")
print(f"Gradient da/dx (using PyTorch): {x.grad.item()}")
