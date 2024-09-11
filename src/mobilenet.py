import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


class MobileNetSSD(nn.Module):
    def __init__(self, num_classes, classification_head, regression_head, extras):
        
        # self.model = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.DEFAULT).features
        self.model = models.vgg16().features
        print(self.model)
        self.num_classes = num_classes
        self.classification_head = classification_head
        self.regression_head = regression_head
        self.extras = extras

    def forward(self, x):
        confidences = []
        locations = []


    def get_params(self):
        pass

# model = MobileNetSSD(None, None, None, None)

model = nn.Sequential(*[models.vgg16().features,
               nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
               nn.Conv2d(1024, 1024, kernel_size=1)
]).cuda()

# print(model)

summary(model, (3, 300, 300))