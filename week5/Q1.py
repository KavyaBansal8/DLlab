import torch
import torch.nn.functional as F

# Original setup
image = torch.rand(6, 6)
image = image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 6, 6)
kernel = torch.ones(3, 3)
kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

# Test stride=2
out_stride2 = F.conv2d(image, kernel, stride=2, padding=0)
print(out_stride2.shape)  # Output shape: torch.Size([1, 1, 2, 2])

# Test padding=1
out_pad1 = F.conv2d(image, kernel, stride=1, padding=1)
print(out_pad1.shape)  # Output shape: torch.Size([1, 1, 6, 6])

# Number of parameters
print(kernel.numel())  # Output: 9