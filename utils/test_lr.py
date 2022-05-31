import torch
import torch.nn as nn

model = nn.Linear(100,10)
sgd = torch.optim.SGD(model.parameters(), 1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sgd, T_max=100, eta_min=1e-7)

for i in range(100):
    lr_scheduler.step(i)
    print(sgd.param_groups[0]['lr'])