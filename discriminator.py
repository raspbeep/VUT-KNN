import torch
import torch.nn as nn

KERNEL_SIZE = 4
PADDING = 1

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.seq = [
            nn.Conv2d(in_channels, out_channels=features[0], kernel_size=KERNEL_SIZE, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_channels = features[0]
        for feature in features[1:]:
            # self.seq.append(Block(in_channels, out_channels=feature, stride=1 if feature == features[-1] else 2))
            self.seq.append(nn.Conv2d(in_channels, out_channels=feature, kernel_size=KERNEL_SIZE, stride=1 if feature == features[-1] else 2, padding=PADDING, bias=True, padding_mode='reflect'))
            self.seq.append(nn.InstanceNorm2d(feature))
            self.seq.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature
        self.seq.append(nn.Conv2d(in_channels, out_channels=1, kernel_size=KERNEL_SIZE, stride=1, padding=1, padding_mode='reflect'))

        self.model: nn.Sequential = nn.Sequential(*self.seq)
        # Apply the initialization
        self.model.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0., std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
    
def test():
    x = torch.randn((1,3,256,256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == '__main__':
    test()
