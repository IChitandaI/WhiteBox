import torch
import torch.nn as nn

def _d_loss(real_logit, fake_logit):
    return 0.5 * (torch.mean((real_logit - 1)**2) + torch.mean(fake_logit**2))

input1 = torch.rand(1, requires_grad=True)
target = torch.empty(1).random_(1)
loss=nn.BCELoss()
print(input1,target)
print(loss(input1,target))
print(_d_loss(input1,target))

