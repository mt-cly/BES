import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50


class Net(nn.Module):

    def __init__(self, frozen=True):
        super(Net, self).__init__()
        self.frozen = frozen
        # set the last layer stride=1 for keeping resolution
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        self.mean_shift = Net.MeanShift(2)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)



        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList(
            [self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

    class MeanShift(nn.Module):

        def __init__(self, num_features):
            super(Net.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        return edge_out

    def trainable_parameters(self):
        return tuple(self.edge_layers.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class Boundary(Net):

    def __init__(self, crop_size=512):
        super(Boundary, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        edge_out = super().forward(x)
        edge_out = torch.sigmoid(edge_out)
        return edge_out

    def infer(self, x):
        """
        for eval and test phase.
        :param x: [image, flipped_image]
        :return: predicted boundary map
        """
        edge_out = super().forward(x)
        edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)
        return edge_out.detach()
