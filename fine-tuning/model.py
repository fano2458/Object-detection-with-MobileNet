import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def get_model():
    return ssdlite320_mobilenet_v3_large(torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)

