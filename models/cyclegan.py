# models.py

import torch
import torch.nn as nn
import functools
import random


###############################################################################
# Helpers
###############################################################################

def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(f"init method {init_type} not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, device):
    net.to(device)
    init_weights(net)
    return net


###############################################################################
# ResNet generator
###############################################################################

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type="reflect", norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        p = 0
        if padding_type == "reflect":
            self.pad = nn.ReflectionPad2d(1)
        elif padding_type == "replicate":
            self.pad = nn.ReplicationPad2d(1)
        elif padding_type == "zero":
            self.pad = nn.Identity()
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block = []
        conv_block += [
            self.pad,
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            self.pad,
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
            norm_layer(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    """
    Classic CycleGAN generator:
    c7s1-64, d128, d256, 9x ResBlocks, u128, u64, c7s1-3
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        assert n_blocks >= 0
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # Downsampling
        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        # Resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type="reflect", norm_layer=norm_layer)]

        # Upsampling
        for i in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
            mult //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


###############################################################################
# PatchGAN discriminator
###############################################################################

class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=True,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=True,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # no sigmoid, use MSELoss

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


###############################################################################
# GAN loss (LSGAN)
###############################################################################

class GANLoss(nn.Module):
    """LSGAN: MSE between prediction and target label"""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_val = 1.0
        else:
            target_val = 0.0
        return torch.full_like(prediction, target_val)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
