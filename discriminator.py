import torch
import torch.nn as nn
import torchvision.models as models

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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG model (e.g., VGG16)
        self.model = models.vgg16(pretrained=True).features.eval()
        # Choose which layers to use for feature extraction
        self.layers = nn.Sequential(*list(self.model.children())[:4])  # Example: Use first 4 layers of VGG16
    
    def forward(self, x):
        # Forward pass through the chosen layers of the pre-trained VGG model
        features = self.layers(x)
        # print(features)
        return features

    # def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
    #     super().__init__()
    #     self.initial = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels=features[0], kernel_size=KERNEL_SIZE, stride=2, padding=1, padding_mode='reflect'),
    #         nn.LeakyReLU(0.2, inplace=True),
    #     )

    #     layers = []
    #     in_channels = features[0]
    #     for feature in features[1:]:
    #         layers.append(Block(in_channels, out_channels=feature, stride=1 if feature == features[-1] else 2))
    #         in_channels = feature
    #     layers.append(nn.Conv2d(in_channels, out_channels=1, kernel_size=KERNEL_SIZE, stride=1, padding=1, padding_mode='reflect'))
    #     self.model = nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.initial(x)
    #     print(torch.sigmoid(self.model(x)))
    #     return torch.sigmoid(self.model(x))
    
def test():
    x = torch.randn((1,3,256,256))
    # model = Discriminator(in_channels=3)
    model = Discriminator()
    # vit_b_32 = models.vit_b_32(pretrained=True)
    # model = models.vgg16(pretrained=True)
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == '__main__':
    test()
