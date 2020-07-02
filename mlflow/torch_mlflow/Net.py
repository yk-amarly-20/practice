# ニューラルネットを定義

import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    """
    ニューラルネット定義
    """

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))

        return F.log_softmax(x)


