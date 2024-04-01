import torch
import torch.nn as nn


# Self-attention mechanism is usually used in parallel with convolutional layers
#   and concatenated before the last layer  
# In this approach the self attention is used after each concolution by using 1x1 
#   conv kernels 
# Inspiration: https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/attention_augmented_conv.py
class AttentionBlock(nn.Module):


    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        
        # Nh != 0 -> Nh is divider cannot be 0 
        # dk % Nh == 0 -> dk should be divided by Nh        
        # dv % Nh == 0 -> dk should be divided by Nh 
        self.dk = 40
        self.dv = 4 
        delf.Nh = 4

        # TODO inst norm and relu should be applied after the attention 





        # TODO maybe consider self.relative mechanism 


class ConvBlock(nn.Module):
    # kwargs - kernel size, stride, padding
    # down -- if True -> then downsampling is done 
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down # 
            # gradient of Conv2d when upsampling
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    # num_residuals =9 if 256**2 or larger, =6 if 128 or smaller
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        # input (B, C, img_size, img_size)
        # output (B, 9, img_size, img_size), 9=num features, number of output channels the convolution produces
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        # input (B, 9, img_size, img_size)
        # output1 (B, 9*2, img_size/2, img_size/2) = (B, 18, 128, 128)
        # output1 (B, 9*4, img_size/4, img_size/4) = (B, 36, 64, 64)
        # Stride 2 is utilizing the pooling 
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        # tanh ensures that the pixels are in range [-1,1]
        return torch.tanh(self.last(x))

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((1, 3, img_size, img_size))
    model = Generator(img_channels, 9)
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == '__main__':
    test()