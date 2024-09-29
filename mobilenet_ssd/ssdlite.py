import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import collections
import itertools
import math


SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.max)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])
        
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors =  torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

image_size = 300

mobilenet_specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2])
]


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True)/6
        return out
    
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x+3, inplace=True)/6
        return out

class relu(nn.Module):
    def forward(self, x):
        out = F.relu(x)
        return out


class SeModule(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//reduction, in_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes),
            hsigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.init_conv2d()
    
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu6(x)
        return x
    
class MBlock(nn.Module):
    def __init__(self, kernel_size, in_planes, expand_planes, out_planes, nolinear, semodule, stride):
        super(MBlock, self).__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_planes, expand_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(expand_planes)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_planes, expand_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.short_cut = nn.Sequential()

        if stride == 1 and in_planes != out_planes:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nolinear1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nolinear2(out)
        if self.se != None:
            out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.short_cut(x) if self.stride == 1 else out
        return out 

class liteConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(liteConv, self).__init__()
        hidden_dim = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.init_conv2d()
    
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu6(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu6(out)
        return out

    
class PredictionConvolutions(nn.Module):
    def __init__(self, class_num, backbone):
        super(PredictionConvolutions, self).__init__()
        self.class_num = class_num
        
        n_boxes = {
            'conv4_3': 4,
            'conv7': 6,
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2': 4
        }
        
        self.loc_conv4_3 = Block(112, n_boxes['conv4_3']*4)
        self.loc_conv7 = Block(960, n_boxes['conv7']*4)
        self.loc_conv8_2 = Block(512, n_boxes['conv8_2']*4)
        self.loc_conv9_2 = Block(256, n_boxes['conv9_2']*4)
        self.loc_conv10_2 = Block(256, n_boxes['conv10_2']*4)
        self.loc_conv11_2 = Block(256, n_boxes['conv11_2']*4)

        self.cl_conv4_3 = Block(112, n_boxes['conv4_3']*class_num)
        self.cl_conv7 = Block(960, n_boxes['conv7']*class_num)
        self.cl_conv8_2 = Block(512, n_boxes['conv8_2']*class_num)
        self.cl_conv9_2 = Block(256, n_boxes['conv9_2']*class_num)
        self.cl_conv10_2 = Block(256, n_boxes['conv10_2']*class_num)
        self.cl_conv11_2 = Block(256, n_boxes['conv11_2']*class_num)
    
    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.class_num)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.class_num)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.class_num)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.class_num)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1 ,self.class_num)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.class_num)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        cls_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)

        return locs, cls_scores

class AuxillaryConvolutions(nn.Module):
    def __init__(self, backbone):
        super(AuxillaryConvolutions, self).__init__()

        self.extras = nn.ModuleList([
            liteConv(960, 512, stride=2),
            liteConv(512, 256, stride=2),
            liteConv(256, 256, stride=2),
            liteConv(256, 256, stride=2),
        ])

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            for layer in c:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(c.bias, 0.)
    
    def forward(self, feats10x10):
        features = []
        x = feats10x10
        for layer in self.extras:
            x = layer(x)
            features.append(x)
        
        features_5x5 = features[0]
        features_3x3 = features[1]
        features_2x2 = features[2]
        features_1x1 = features[3]
        return features_5x5, features_3x3, features_2x2, features_1x1


class SSDLite(nn.Module):
    def __init__(self, class_num, backbone, device):
        super(SSDLite, self).__init__()
        self.class_num = class_num
        self.backbone = backbone
        
        priors = generate_ssd_priors(mobilenet_specs, image_size)

        self.priors = torch.FloatTensor(priors).to(device)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish(True)
        self.last_conv = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.last_bn = nn.BatchNorm2d(960)
        self.last_hs = nn.Hardswish(True)
        # self.base_net = MobileNetV3_Large(class_num)
        self.bneck = nn.Sequential(
            MBlock(3, 16, 16, 16, relu(), None, 1),
            MBlock(3, 16, 64, 24, relu(), None, 2),
            MBlock(3, 24, 72, 24, relu(), None, 1),
            MBlock(5, 24, 72, 40, relu(), SeModule(72), 2),
            MBlock(5, 40, 120, 40, relu(), SeModule(120), 1),
            MBlock(5, 40, 120, 40, relu(), SeModule(120), 1),
            MBlock(3, 40, 240, 80, hswish(), None, 2),
            MBlock(3, 80, 200, 80, hswish(), None, 1),
            MBlock(3, 80, 184, 80, hswish(), None, 1),
            MBlock(3, 80, 184, 80, hswish(), None, 1),
            MBlock(3, 80, 480, 112, hswish(), SeModule(480), 1),
            MBlock(3, 112, 672, 112, hswish(), SeModule(672), 1),
            MBlock(5, 112, 672, 160, hswish(), SeModule(672), 2),
            MBlock(5, 160, 960, 160, hswish(), SeModule(960), 1),
            MBlock(5, 160, 960, 160, hswish(), SeModule(960), 1),
        )

        self.aux_net = AuxillaryConvolutions(backbone=self.backbone)
        self.prediction_net = PredictionConvolutions(class_num=self.class_num, backbone=self.backbone)
    
    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hs1(x)
        for index, feat in enumerate(self.bneck):
            x = feat(x)
            if index == 10:
                features_19x19 = x
            if index == 14:
                x = self.last_conv(x)
                x = self.last_bn(x)
                x = self.last_hs(x)
                features_10x10 = x

        features_5x5, features_3x3, features_2x2, features_1x1 = self.aux_net(x)

        locs, cls_scores = self.prediction_net(features_19x19, features_10x10, features_5x5, features_3x3, features_2x2, features_1x1)

        return locs, cls_scores


from torchsummary import summary

if __name__ == '__main__':

    model = SSDLite(class_num=2, backbone='MobileNetV3_Large', device='cuda').cuda()
    summary(model, (3, 300, 300))
    