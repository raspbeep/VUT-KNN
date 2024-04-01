import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-attention mechanism is usually used in parallel with convolutional layers
#   and concatenated before the last layer  
# In this approach the self attention is used after each concolution by using 1x1 
#   conv kernels 
# Inspiration: https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/attention_augmented_conv.py
class AttentionBlock(nn.Module):
    

    def __init__(self, in_channels, activation):
        super(AttentionBlock, self).__init__()
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels, in_channels//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels , in_channels, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, x):
        # inputs :
        #     x : input feature maps( B X C X W X H)
        # returns :
        #     out : self attention value + input feature 
        #     attention: B X N X N (N is Width*Height)
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        
        return out,attention 


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
        self.attn1 = AttentionBlock(3, 'relu')
        self.attn2 = AttentionBlock(6, 'relu')
        self.attn3 = AttentionBlock(12, 'relu')
        
        self.attn11 = AttentionBlock(128, 'relu')
        self.attn21 = AttentionBlock(64, 'relu')

    def forward(self, x):
        
        x = self.initial(x)
        #x, p1 = self.attn1(x)
        x = self.down_blocks[0](x)
        #x, p2 = self.attn2(x)
        x = self.down_blocks[1](x)
        #x, p3 = self.attn3(x)

        for block in self.residual_blocks:
            x = block(x)
        
        x = self.up_blocks[0](x)
        x = self.up_blocks[1](x)

        #print(self.attn11)
        #print(self.last)
        #print(x.size())
        x, p = self.attn21(x)
        #print(x.size())
        
        return torch.tanh(self.last(x)), p
        ## P can be returned and optimized 
        #return torch.tanh(self.last(x)), p

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