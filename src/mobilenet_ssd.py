import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary


class MobileNetSSD(nn.Module):
    # TODO change everything to power of 2
    # input size is 640 by 640
    def __init__(self, n_classes):
        super(MobileNetSSD, self).__init__()
        self.n_classes = n_classes

        self.base_model = models.mobilenet_v3_small().features
        self.conv1_1 = nn.Conv2d(576, 256, kernel_size=1, padding=0) 
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        n_boxes = {'base': 4,
                   'conv1': 6,
                   'conv2': 6,
                   'conv3': 6
                   }

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_base = nn.Conv2d(576, n_boxes['base'] * 4, kernel_size=3, padding=1)
        self.loc_conv1 = nn.Conv2d(256, n_boxes['conv1'] * 4, kernel_size=3, padding=1)
        self.loc_conv2 = nn.Conv2d(256, n_boxes['conv2'] * 4, kernel_size=3, padding=1)
        self.loc_conv3 = nn.Conv2d(256, n_boxes['conv3'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_base = nn.Conv2d(576, n_boxes['base'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv1 = nn.Conv2d(256, n_boxes['conv1'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv2 = nn.Conv2d(256, n_boxes['conv2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv3 = nn.Conv2d(256, n_boxes['conv3'] * n_classes, kernel_size=3, padding=1)


    def forward(self, x):
        batch_size = x.shape[0]
        # TODO consider different activations
        x = self.base_model(x)                      # (B, 576, 20, 20)
        base_feats = x

        x = F.relu(self.conv1_1(x))                 # (B, 256, 20, 20)
        x = F.relu(self.conv1_2(x))                 # (B, 256, 10, 10)
        conv1_feats = x

        x = F.relu(self.conv2_1(x))                 # (B, 128, 10, 10)
        x = F.relu(self.conv2_2(x))                 # (B, 256, 5, 5)
        conv2_feats = x

        x = F.relu(self.conv3_1(x))                 # (B, 128, 5, 5)
        x = F.relu(self.conv3_2(x))                 # (B, 256, 3, 3)
        conv3_feats = x

        # locations
        l_base = self.loc_base(base_feats)          # (B, 16, 20, 20)
        l_base = l_base.permute(0, 2, 3, 1).contiguous()
        l_base = l_base.view(batch_size, -1, 4)     # (B, 1600, 4)
        
        l_conv1 = self.loc_conv1(conv1_feats)       # (B, 24, 10, 10)
        l_conv1 = l_conv1.permute(0, 2, 3, 1).contiguous()
        l_conv1 = l_conv1.view(batch_size, -1, 4)   # (B, 600 4)

        l_conv2 = self.loc_conv2(conv2_feats)       # (B, 24, 5, 5)
        l_conv2 = l_conv2.permute(0, 2, 3, 1).contiguous()
        l_conv2 = l_conv2.view(batch_size, -1, 4)   # (B, 150, 4)

        l_conv3 = self.loc_conv3(conv3_feats)       # (B, 24, 3, 3)
        l_conv3 = l_conv3.permute(0, 2, 3, 1).contiguous()
        l_conv3 = l_conv3.view(batch_size, -1, 4)   # (B, 54, 4)
    
        # classes
        c_base = self.cl_base(base_feats)                      
        c_base = c_base.permute(0, 2, 3, 1).contiguous()
        c_base = c_base.view(batch_size, -1, self.n_classes)   

        c_conv1 = self.cl_conv1(conv1_feats)                   
        c_conv1 = c_conv1.permute(0, 2, 3, 1).contiguous()
        c_conv1 = c_conv1.view(batch_size, -1, self.n_classes) 

        c_conv2 = self.cl_conv2(conv2_feats)                   
        c_conv2 = c_conv2.permute(0, 2, 3, 1).contiguous()
        c_conv2 = c_conv2.view(batch_size, -1, self.n_classes) 

        c_conv3 = self.cl_conv3(conv3_feats)                   
        c_conv3 = c_conv3.permute(0, 2, 3, 1).contiguous()
        c_conv3 = c_conv3.view(batch_size, -1, self.n_classes) 

        locs = torch.cat([l_base, l_conv1, l_conv2, l_conv3], dim=1)
        classes_scores = torch.cat([c_base, c_conv1, c_conv2, c_conv3], dim=1)

        return locs, classes_scores

model = MobileNetSSD(n_classes=1).cuda().eval()

summary(model, (3, 640, 640))

input = torch.rand([1, 3, 640, 640]).cuda()

with torch.no_grad():
    locs, classes_scores = model(input)
    print(locs.shape)
    print(classes_scores.shape)
