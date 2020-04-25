import os

import numpy as np
import torch
import torch.nn as nn


class Residual(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
        bns = [layer for layer in self.modules() if isinstance(layer, nn.BatchNorm2d)]
        last_bn = bns[-1]
        last_bn.weight.data.zero_()

    def forward(self, input):
        return input + super().forward(input)


def conv_block(c_in, c_out, ks=3, stride=1, activation=True):
    pad = (ks - 1) // 2  # preserve spatial dimension
    non_linearity = nn.ReLU() if activation else nn.Identity()
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, ks, stride, pad, bias = False),
        nn.BatchNorm2d(c_out),
        non_linearity,
    )


def residual_block(c_in, c_out, ks=3, stride=1, repeat=None):
    if repeat is None:
        return nn.Sequential(
            Residual(
                conv_block(c_in, c_out, ks, stride),
                conv_block(c_out, c_out, ks, stride, activation=False),
            ),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            *[residual_block(c_in, c_out, ks=3, stride=1) for _ in range(repeat)]
        )


def downsampling_block(c_in, c_out, ks=3):
    return nn.Sequential(
        conv_block(c_in, c_out, ks, 2),
        conv_block(c_out, c_out, ks, 1),
    )


def define_resnet34(n_classes = 120):
    return nn.Sequential(
        conv_block(3, 64, 3, 1),
        downsampling_block(64, 64),
        downsampling_block(64, 64),
        residual_block(64, 64, repeat=3),
        downsampling_block(64, 128),
        residual_block(128, 128, repeat=3),
        downsampling_block(128, 256),
        residual_block(256, 256, repeat=5),
        downsampling_block(256, 512),
        residual_block(512, 512, repeat=2),
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(512, n_classes),
    )

model = define_resnet34()
