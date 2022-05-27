from utils.registry import Registry
import torch.nn as nn



class Operation(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation = None

    def forward(self, x):
        return self.operation(x)



class Shortcut(Operation):
    def __init__(self, is_downsampling, in_channels=None, out_channels=None, stride=2,version=1, drop_path=nn.Identity()):
        super().__init__()
        self.identity = None
        self.drop_path = drop_path
        if is_downsampling:
            if version == 1:
                self.operation = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
            elif version == 2:
                self.operation = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )

            else:
                raise ValueError(f'Version Shortcut {version} does not exist')
        else:
            self.operation = nn.Identity()

    def forward(self, x):
        return self.drop_path(x).add(self.operation(self.identity))
