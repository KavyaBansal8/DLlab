import torch

x = torch.randn(3, 4)
print(x)
indices = torch.tensor([0, 2])
print(torch.index_select(x, 0, indices))
print(torch.index_select(x, 1, indices))