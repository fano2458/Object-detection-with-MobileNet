from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
transform = transforms.Compose([
    transforms.ToTensor(),
])


def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # transform the image to tensor
    image = transform(image).to(device)
    # add a batch dimension
    image = image.unsqueeze(0) 
    # get the predictions on the image
    with torch.no_grad():
        outputs = model(image) 
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in labels.cpu().numpy()]
    return boxes, pred_classes, labels


def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image


def get_model(num_classes=None):
    model = ssdlite320_mobilenet_v3_large(torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    return model.eval().to(device)


model = get_model()

# read the image
image = Image.open("/home/fano/Desktop/CSCI 494/final_project/Object-detection-with-MobileNet/datasets/open-images-bus-trucks/images/0a7fdbbd74c42e18.jpg")
# detect outputs
boxes, classes, labels = predict(image, model, device, 0.5) # threshold
# draw bounding boxes
image = draw_boxes(boxes, classes, labels, image)
cv2.imshow('Image', image)
cv2.waitKey(0)
