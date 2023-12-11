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
    def __init__(self, input_dim, output_dim, padding, pool=True):
        super().__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        # layers.append(nn.BatchNorm2d(output_dim))  # stabilize training
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        # layers.append(nn.BatchNorm2d(output_dim))  # stabilize training
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        # layers.append(nn.BatchNorm2d(output_dim))  # stabilize training
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(lambda m: init_weights(m, init="kaiming"))

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, padding):
        super().__init__()

        self.conv_block = DownConvBlock(input_dim, output_dim, padding, pool=False)

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

    def __init__(self, 
                 num_input_channels,
                 num_channels,
                 num_classes,
                 apply_last_layer=True, 
                 padding=True):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.padding = padding
        self.apply_last_layer = apply_last_layer

        # Encoder
        self.encoder = nn.ModuleList()

        for i, n_channels in enumerate(self.num_channels):

            if i == 0:
                pool = False
            else:
                pool = True

            in_channels = self.num_input_channels if i == 0 else self.num_channels[i-1]
            out_channels = n_channels

            self.encoder.append(
                DownConvBlock(in_channels, 
                              out_channels, 
                              padding, 
                              pool=pool, 
                              ))

        # Decoder
        self.decoder = nn.ModuleList()

        n = len(self.num_channels) - 2
        for i in range(n, -1, -1):
            in_channels = out_channels + self.num_channels[i]
            out_channels = self.num_channels[i]
            self.decoder.append(
                UpConvBlock(
                    in_channels, 
                    out_channels, 
                    padding
                    ))

        # Last layer
        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            #nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            #nn.init.normal_(self.last_layer.bias)
            #self.last_layer.apply(lambda m: init_weights(m, init="kaiming"))
        

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder)-1:
                blocks.append(x)

        for i, up in enumerate(self.decoder):
            x = up(x, blocks[-i-1])

        del blocks
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x

