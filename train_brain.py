
import os
import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch.nn as nn
import torch.nn.functional as F
import torch

import multiscale_constr_model

class MultiChannelConvolutionalMatrixNet(networks.ConvolutionalMatrixNet):
    def __init__(self, dimension=2, channels=1):
        super().__init__()
        self.dimension = dimension

        if dimension == 2:
            self.Conv = nn.Conv2d
            self.avg_pool = F.avg_pool2d
        else:
            self.Conv = nn.Conv3d
            self.avg_pool = F.avg_pool3d

        self.features = [2 * channels, 16, 32, 64, 128, 256, 512]
        self.convs = nn.ModuleList([])
        for depth in range(len(self.features) - 1):
            self.convs.append(
                self.Conv(
                    self.features[depth],
                    self.features[depth + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
        self.dense2 = nn.Linear(512, 300)
        self.dense3 = nn.Linear(300, 6 if self.dimension == 2 else 12)
        torch.nn.init.zeros_(self.dense3.weight)
        torch.nn.init.zeros_(self.dense3.bias)



BATCH_SIZE = 1
GPUS = 4


def make_batch(dataset):
    image = torch.cat([random.choice(dataset)[None] for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

input_shape = [1, 1, 140, 140, 140]
def make_net(lmbda=5):

    net = multiscale_constr_model.FirstTransform(
        multiscale_constr_model.TwoStepInverseConsistent(
            multiscale_constr_model.ConsistentFromMatrix(
                MultiChannelConvolutionalMatrixNet(dimension=3, channels=4)
            ),
            multiscale_constr_model.TwoStepInverseConsistent(
                multiscale_constr_model.ConsistentFromMatrix(
                    MultiChannelConvolutionalMatrixNet(dimension=3, channels=4)
                ),
                multiscale_constr_model.TwoStepInverseConsistent(
                    multiscale_constr_model.ICONSquaringVelocityField(
                        networks.tallUNet2(dimension=3, input_channels=4)
                    ),
                    multiscale_constr_model.ICONSquaringVelocityField(
                        networks.tallUNet2(dimension=3, input_channels=4)
                    ),
                ),
            ),
        )
    )

    loss = multiscale_constr_model.VelocityFieldDiffusion(net, icon.LNCC(5), lmbda)
    loss.assign_identity_map(input_shape)
    return loss


if __name__ == "__main__":
    footsteps.initialize()

    dataset = torch.load("results/preprocessed_tensor/train_tensor.trch")

    batch_function = lambda: (make_batch(dataset), make_batch(dataset))

    loss = make_net()

    net_par = torch.nn.DataParallel(loss).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(net_par, optimizer, batch_function, unwrapped_net=loss)
