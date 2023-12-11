"""
UNet implementation taken from URL
"""
import torch
import torch.nn as nn

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m, init="kaiming"):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        if init == "kaiming":
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.normal_(m.weight, std=0.001)
            #nn.init.normal_(m.bias, std=0.001)
            truncated_normal_(m.bias, mean=0, std=0.001)
        elif init == "orthogonal_normal":
            nn.init.orthogonal_(m.weight)
            truncated_normal_(m.bias, mean=0, std=0.001)
            #nn.init.normal_(m.bias, std=0.001)
        else:
            raise Exception("Undefined weight initialization")

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, padding, pool=True, init=None):
        super().__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.BatchNorm2d(output_dim))  # stabilize training
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.BatchNorm2d(output_dim))  # stabilize training
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.BatchNorm2d(output_dim))  # stabilize training
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        if init:
            self.layers.apply(lambda m: init_weights(m, init=init))

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, padding, init=None):
        super().__init__()

        self.conv_block = DownConvBlock(input_dim, output_dim, padding, pool=False, init=init)

    def forward(self, x, bridge):
        up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        
        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out

class UNet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    batch_norm (bool): aplies batch norm between convolutional blocks
    dropout (float): sets dropout rate
    init (str): init type (orthonormal, kaiming)
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, batch_norm=True, dropout=None, init=None, apply_last_layer=True, padding=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, padding, pool=pool, init=init))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, padding, init=init))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            #nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            #nn.init.normal_(self.last_layer.bias)
            if init:
                self.last_layer.apply(lambda m: init_weights(m, init=init))
        

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x

# Basic UNet model

# class down_conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()    
#         self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.a1 = nn.ReLU()        
#         self.c2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.a2 = nn.ReLU()        

#     def forward(self, x):
#         out = self.c1(x)
#         out = self.bn1(out)
#         out = self.a1(out)
#         out = self.c2(out)
#         out = self.bn2(out)
#         out = self.a2(out)
#         return out
    
# class up_conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels):
#         super().__init__()    
#         self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(in_channels//2)
#         self.a1 = nn.ReLU()
#         self.c2 = nn.Conv2d(in_channels=in_channels//2 + mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.a2 = nn.ReLU()       
#         self.c3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.a3 = nn.ReLU()        

#     def forward(self, x, x_down):
#         out = self.c1(x)
#         out = self.bn1(out)
#         out = self.a1(out)
#         out = torch.cat([x_down, out], dim=1)
#         out = self.c2(out)
#         out = self.bn2(out)
#         out = self.a2(out)
#         out = self.c3(out)
#         out = self.bn3(out)
#         out = self.a3(out)
#         return out

# class UNet(nn.Module):
    # def __init__(self, num_classes=2, num_levels=4):
    #     super().__init__()   
    #     self.num_classes = num_classes
    #     self.num_levels = num_levels

    #     self.in_block = down_conv_block(in_channels=1, out_channels=64)

    #     self.down_blocks = []
    #     self.down_lvs = []
    #     for lv in range(num_levels-1):
    #         in_channels = 2**(6+lv)
    #         out_channels = 2**(7+lv)
    #         self.down_lvs.append((in_channels, out_channels))
    #         self.down_blocks.append(down_conv_block(in_channels=in_channels, out_channels=out_channels))
    #     self.down_blocks = nn.ModuleList(self.down_blocks)

    #     self.bottom_block = down_conv_block(in_channels=2**(6 + self.num_levels - 1), out_channels=2**(6 + self.num_levels))

    #     self.up_blocks = []
    #     self.up_lvs = []
    #     for lv in reversed(range(num_levels-1)):
    #         in_channels = 2**(7+lv+1)
    #         mid_channels = in_channels // 2
    #         out_channels = mid_channels
    #         self.up_lvs.append((in_channels, out_channels, mid_channels))
    #         self.up_blocks.append(up_conv_block(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels))
    #     self.up_blocks = nn.ModuleList(self.up_blocks)
        
    #     self.interp = lambda input: nn.functional.interpolate(input, scale_factor=2)
    #     self.pool = nn.MaxPool2d(kernel_size=2)

    #     # classification layer
    #     self.cl = nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)

    # def forward(self, x):              

    #     out = self.in_block(x)  # [64, H, W]

    #     # Downward path
    #     down_outs = [] 
    #     for db in self.down_blocks:
    #         out = db(out)  # feature propagation
    #         down_outs.append(out)
    #         out = self.pool(out)  # feature aggregation

    #     # Bottom
    #     out = self.bottom_block(out)

    #     # Upward path
    #     up_outs = []
    #     for dbo, ub in zip(reversed(down_outs), self.up_blocks):
    #         out = self.interp(out)  # feature dis-aggregation
    #         out = ub(out, dbo)  # feature propagation
    #         up_outs.append(out)
        
    #     out = self.cl(out)  # [15, H, W]
    #     return out