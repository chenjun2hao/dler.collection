from torch import nn
import torch
from thop import profile

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == "__main__":
    x = torch.rand(4, 256, 200, 200)
    net = SELayer(256)
    y = net(x)
    print(y.shape)
    macs, params = profile(net, inputs=(x, ))           # 40993792.0 8192.0
    print(macs, params)