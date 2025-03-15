"""
     build a UNet model to segment images.
"""

# Import Libraries ====================================
import torch
from torch import nn
import config


def conv3x3block(in_channels, out_channels, padding, batch_norm):
    """
    A module that performs 2 convolutions. A batch normalisation (if 'True')
    and a ReLU activation follows each convolution.
    [conv -> BN -> ReLU] -> [conv -> BN -> ReLU]
    :param in_channels:
    :param out_channels:
    :param padding:
    :param batch_norm:
    :return:
    """
    layers = []
    layers.append(conv3x3(in_channels=in_channels, out_channels=out_channels, padding=padding))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    layers.append(conv3x3(in_channels=out_channels, out_channels=out_channels, padding=padding))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, padding):
    """ Conv2d with 3x3 kernel, stride=1 """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding)


def up_conv2x2(in_channels, out_channels):
    """ ConvTranspose2d with 2x2 kernel, stride=2 """
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)


def conv1x1(in_channels, out_channels):
    """ Conv2d with 1x1 kernel, stride=1 """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        return torch.cat([layer_1, layer_2], dim=1)


class UNet(nn.Module):
    """ U-Net class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    - padding is used in 3x3 convolutions to prevent loss
        of border pixels
    - merging outputs does not require cropping due to (1)
    """

    def __init__(self, in_channels=1, nb_classes=2, init_features=64, padding='same', batch_norm=False, dropout=False):
        """
        Arguments:
        :param in_channels: int, number of channels in the input tensor.
                 Default is 1 for single-grayscale inputs (like original paper). Need 3 for RGB images.
        :param nb_classes: int, number of different classes that defines the number of output channels
                (in classification/semantic segmentation). Default: 2.
        :param init_features: int, number of filters for the first convolution layer.
        :param padding: str, Padding mode of convolutions. Choices:
            - 'same' (default): Use SAME-convolutions in every layer:
              zero-padding inputs so that all convolutions preserve spatial
              shapes.
            - 'valid' : Use VALID-convolutions in every layer: no padding is
              used, so every convolution layer reduces spatial shape by 2 in
              each dimension.
        :param batch_norm: bool, batch normalization that should be applied at the end
            of each block. Note that it is applied before the activation conv
            layers, not before the activation. Default : false for no batch normalization (original paper).
        :param dropout: bool, if 'True', performs dropout after each
            pool in the network. Default : False.
        """
        super(UNet, self).__init__()

        features = init_features
        self.dropout = dropout

        # Display model parameters if True
        if config.VERBOSE:
            print('---------------------------------')
            print('Nb channels          : ', in_channels)
            print('Nb Classes           : ', nb_classes)
            print('Nb features          : ', init_features)
            print('Padding              : ', padding)
            print('BatchNormalisation   : ', batch_norm)
            print('Dropout              : ', dropout)
            print('---------------------------------')

        # Input: image.height x image.width x in_channels (pour padding = 'same')
        self.conv1 = conv3x3block(in_channels=in_channels, out_channels=features, padding=padding,
                                  batch_norm=batch_norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if dropout:
            self.drop1 = nn.Dropout(0.25)

        # Input: image.height/2 x image.width/2 x features
        self.conv2 = conv3x3block(in_channels=features, out_channels=features * 2, padding=padding,
                                  batch_norm=batch_norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if dropout:
            self.drop2 = nn.Dropout(0.5)

        # Input: image.height/4 x image.width/4 x features x 2
        self.conv3 = conv3x3block(in_channels=features * 2, out_channels=features * 4, padding=padding,
                                  batch_norm=batch_norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        if dropout:
            self.drop3 = nn.Dropout(0.5)

        # Input: image.height/8 x image.width/8 x features x 4
        self.conv4 = conv3x3block(in_channels=features * 4, out_channels=features * 8, padding=padding,
                                  batch_norm=batch_norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        if dropout:
            self.drop4 = nn.Dropout(0.5)

        # Input: image.height/16 x image.width/16 x features x 8
        self.bottleneck = conv3x3block(in_channels=features * 8, out_channels=features * 16, padding=padding,
                                       batch_norm=batch_norm)

        # Decoder
        self.up_conv4 = up_conv2x2(features * 16, features * 8)
        if dropout:
            self.udrop4 = nn.Dropout(0.5)
        self.conv5 = conv3x3block(in_channels=features * 8 * 2, out_channels=features * 8, padding=padding,
                                  batch_norm=batch_norm)
        self.up_conv3 = up_conv2x2(features * 8, features * 4)
        if dropout:
            self.udrop3 = nn.Dropout(0.5)
        self.conv6 = conv3x3block(in_channels=features * 4 * 2, out_channels=features * 4, padding=padding,
                                  batch_norm=batch_norm)
        self.up_conv2 = up_conv2x2(features * 4, features * 2)
        if dropout:
            self.udrop2 = nn.Dropout(0.5)
        self.conv7 = conv3x3block(in_channels=features * 2 * 2, out_channels=features * 2, padding=padding,
                                  batch_norm=batch_norm)
        self.up_conv1 = up_conv2x2(features * 2, features)
        if dropout:
            self.udrop1 = nn.Dropout(0.5)
        self.conv8 = conv3x3block(in_channels=features * 2, out_channels=features, padding=padding,
                                  batch_norm=batch_norm)
        self.conv_final = conv1x1(in_channels=features, out_channels=nb_classes)
        self.concat = Concatenate()

    def forward(self, x):
        # Contracting path
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        if self.dropout:
            p1 = self.drop1(p1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        if self.dropout:
            p2 = self.drop2(p2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        if self.dropout:
            p3 = self.drop3(p3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        if self.dropout:
            p4 = self.drop4(p4)

        center = self.bottleneck(p4)

        # Expansive path : decoder
        u4 = self.up_conv4(center)
        u4 = self.concat(u4, c4)
        if self.dropout:
            u4 = self.udrop4(u4)
        c5 = self.conv5(u4)

        u3 = self.up_conv3(c5)
        u3 = self.concat(u3, c3)
        if self.dropout:
            u3 = self.udrop3(u3)
        c6 = self.conv6(u3)

        u2 = self.up_conv2(c6)
        u2 = self.concat(u2, c2)
        if self.dropout:
            u2 = self.udrop2(u2)
        c7 = self.conv7(u2)

        u1 = self.up_conv1(c7)
        u1 = self.concat(u1, c1)
        if self.dropout:
            u1 = self.udrop1(u1)
        c8 = self.conv8(u1)

        # No softmax is used, so you need to apply it in the loss.
        out = self.conv_final(c8)

        return out
