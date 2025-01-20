import torch

tensor= torch.randint(1,10,(7,7))
print(tensor)

tensor2= torch.randint(1,10,(1,7))
print(tensor2)

tensor2=torch.transpose(tensor2,0,1)
print(tensor2)

multiplied= torch.matmul(tensor,tensor2)
print(multiplied)
