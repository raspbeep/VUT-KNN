import torch
import torch.nn as nn
import torchvision.models as models

KERNEL_SIZE = 4
PADDING = 1

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG model (e.g., VGG16)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Choose which layers to use for feature extraction
        self.layers = nn.Sequential(*list(self.model.children())[:5])  # Example: Use first 4 layers of VGG16
    
    def forward(self, x):
        # Forward pass through the chosen layers of the pre-trained VGG model
        features = self.layers(x)
        # print(features)
        return features
    
def test():
    x = torch.randn((1,3,256,256))
    # model = Discriminator(in_channels=3)
    model = Discriminator()
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == '__main__':
    test()