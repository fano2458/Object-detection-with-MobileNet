from torchvision import models
from torchsummary import summary

model = models.mobilenet_v3_large(models.MobileNet_V3_Large_Weights.DEFAULT).features.cuda()
summary(model, (3, 640, 640))