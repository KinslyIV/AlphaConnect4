import torch.nn as nn
import torch.nn.functional as F
from connectx import Game


class ResNet(nn.Module):
    def __init__(self, game : Game, device, struct: list[int] = [64 for i in range(8)]):
        super().__init__()

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, struct[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(struct[0]),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(struct[i], size) for i, size in enumerate(struct[1:])]
        )

        self.backBone.insert(0, ResBlock(struct[0], struct[0]))
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(struct[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rows * game.columns, game.action_size),
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(struct[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rows * game.columns, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
    
class ResBlock(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.conv1 = nn.Conv2d(size_in, size_in, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(size_in)
        self.conv2 = nn.Conv2d(size_in, size_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(size_out)

        # self.projection = None
        # if size_in != size_out:
        #     self.projection = nn.Conv2d(size_in, size_out, kernel_size=1)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # if self.projection is not None:
        #     residual = self.projection(residual)
        x += residual
        x = F.relu(x)
        return x