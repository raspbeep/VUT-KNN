import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL_SIZE = 4
PADDING = 1

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=stride, padding=PADDING, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.shape
        query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width*height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        out = self.gamma*out + x
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=features[0], kernel_size=KERNEL_SIZE, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = SelfAttention(features[0])

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, out_channels=feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, out_channels=1, kernel_size=KERNEL_SIZE, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.attention(x)
        return torch.sigmoid(self.model(x))
    
def test():
    x = torch.randn((1,3,256,256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == '__main__':
    test()
