import torch
import torch.nn as nn
import torch.nn.functional as F


# Self-attention mechanism is usually used in parallel with convolutional layers
#   and concatenated before the last layer  
# In this approach the self attention is used after each concolution by using 1x1 
#   conv kernels 
# Inspiration: https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/attention_augmented_conv.py
class AttentionBlock(nn.Module):
    
    
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, stride=1, use_act=True):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = 1

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."


        #self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels - self.dv, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.InstanceNorm2d(out_channels - self.dv),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)


#    def __init__(self, in_channels, out_channels, ker_size, down=True, use_act=True, **kwargs):
#        super().__init__()
        
#        # Nh != 0 -> Nh is divider cannot be 0 
#        # dk % Nh == 0 -> dk should be divided by Nh        
#        # dv % Nh == 0 -> dk should be divided by Nh 
#        self.dk = 40 
#        self.dv = 4 
#        self.Nh = 4
        
#        # Normal convolution of the conv block 
#        self.conv = nn.Conv2d(in_channels, out_channels - self.dk, padding_mode='reflect', kernel_size=ker_size, **kwargs)
##        nn.Sequential(
##            nn.Conv2d(in_channels, out_channels - self.dv, padding_mode='reflect', **kwargs)
##            if down # 
##            # gradient of Conv2d when upsampling
##            else nn.ConvTranspose2d(in_channels, out_channels -self.dv, **kwargs),
##            nn.InstanceNorm2d(out_channels - self.dv),
##            nn.ReLU(inplace=True) if use_act else nn.Identity()
##        )
        
#        # querry key and value 
#        self.qkv_conv       = nn.Conv2d(in_channels, 2* self.dk + self.dv, padding_mode='reflect', kernel_size=ker_size, **kwargs)
        
#        # Attention itself is realized by using kernels of size one 
#        #  which will efectively extract important features 
#        self.attn_out = nn.Conv2d(in_channels, self.dk, kernel_size=1, **kwargs)

#        # TODO maybe consider self.relative mechanism 


#    def forward(self, x):
        
#        # Output of the convolutional layer 
#        conv_out = self.conv(x)
        
#        batch, _, height, width = conv_out.size()
        
#        # flat_q, flat_k, flat_v
#        # (batch_size, Nh, height * width, dvh or dkh)
#        # dvh = dv / Nh, dkh = dk / Nh
#        # q, k, v
#        # (batch_size, Nh, height, width, dv or dk)
#        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
#        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
       
#        weights = F.softmax(logits, dim=-1)

#        # attn_out
#        # (batch, Nh, height * width, dvh)
#        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
#        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
#        # combine_heads_2d
#        # (batch, out_channels, height, width)
#        attn_out = self.combine_heads_2d(attn_out)
#        attn_out = self.attn_out(x)

#        # Finally concatenate convolution and attention 
#        combine = torch.cat((conv_out, attn_out), dim=1)

#        return combine


    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


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
            #ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            #ConvBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1)
            
            AttentionBlock(channels, channels, kernel_size=3, dk=40, dv=4, Nh=4, stride=1),
            AttentionBlock(channels, channels, kernel_size=3, dk=40, dv=4, Nh=4, stride=1),
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
                #ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                #ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
                
                AttentionBlock(num_features, num_features*2, kernel_size=3, dk=40, dv=4, Nh=4, stride=2),
                AttentionBlock(num_features*2, num_features*4, kernel_size=3, dk=40, dv=4, Nh=4, stride=2),
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
                #AttentionBlock(num_features*4, num_features*2, kernel_size=3, dk=40, dv=4, Nh=4, stride=2),
                #AttentionBlock(num_features*4, num_features*2, kernel_size=3, dk=40, dv=4, Nh=4, stride=2),
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