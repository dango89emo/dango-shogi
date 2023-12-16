import torch
import torch.nn as nn
import torch.nn.functional as F
from pythondlshogi.common import MOVE_DIRECTION_LABEL_NUM


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        ch = 192
        num_layer = 13
        layers = [
            nn.Conv2d(in_channels=104 if i == 0 else ch, out_channels=ch, kernel_size=3, padding=1) for i in range(num_layer-1)
            ]
        self.layers = nn.ModuleList(layers)
        self.l_last = nn.Conv2d(in_channels=ch, out_channels=MOVE_DIRECTION_LABEL_NUM, kernel_size=1, bias=False)
        self.l_last_bias = nn.Parameter(torch.zeros(9 * 9 * MOVE_DIRECTION_LABEL_NUM))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.l_last(x)
        x = x.view(-1, 9 * 9 * MOVE_DIRECTION_LABEL_NUM)
        return x + self.l_last_bias
