import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

KERNEL_SIZE = 4
PADDING = 1

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_32(pretrained=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
        ])

    def forward(self, x):
        # Resize input images
        x = self.transform(x)
        # Forward pass through the ViT model
        x = self.vit(x)
        return x
    
def test():
    x = torch.randn((1,3,256,256))
    # model = Discriminator(in_channels=3)
    model = Discriminator()
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == '__main__':
    test()
