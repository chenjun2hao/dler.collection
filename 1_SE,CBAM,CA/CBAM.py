import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),       # 256*16
                               nn.BatchNorm2d(in_planes // 16),                             # 只有weight，bias两个可学习参数
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))        # 16*256
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)       # 2*1*7*7
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x1 = self.ca(x) * x
        x2 = self.sa(x1) * x1
        return x2


if __name__ == "__main__":
    x = torch.rand(4, 256, 200, 200)
    net = CBAM(256)
    y = net(x)
    print(y.shape)
    macs, params = profile(net, inputs=(x, ))       # 浮点运算数, 参数量：56706560.0 8290.0
    print(macs, params)
    # summary(net, (256, 200, 200), batch_size=4, device="cpu")