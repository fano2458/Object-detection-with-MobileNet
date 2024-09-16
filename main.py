import pandas as pd
import torch

import collections, os, torch
from PIL import Image
from dataset import OpenDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy as np
import os
import math
import glob
from tqdm import tqdm

# trn_ids, val_ids = train_test_split(df.ImageID.unique(), test_size=0.2, random_state=99)
# # trn_ids, val_ids = train_test_split(trn_ids, train_size=0.8, random_state=99)
# trn_df, val_df = df[df['ImageID'].isin(trn_ids)], df[df['ImageID'].isin(val_ids)]
# print(len(trn_df), len(val_df))

# train_ds = OpenDataset(trn_df)
# test_ds = OpenDataset(val_df)

# train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True, shuffle=True, pin_memory=False)
# test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True, shuffle=False, pin_memory=False)

train_loader = None
test_loader = None
num_classes = None


def train_batch(inputs, model, criterion, optimizer):
    model.train()
    N = len(train_loader)
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    

@torch.no_grad()
def validate_batch(inputs, model, criterion):
    model.eval()
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    return loss


from src.mobilenet_ssd import MultiBoxLoss , MobileNetSSD
from src.detect import *

n_epochs = 15

model = MobileNetSSD(num_classes).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

for epoch in range(n_epochs):
    # adjust_learning_rate(optimizer=optimizer, epoch=epoch)
    _n = len(train_loader)
    for inputs in tqdm(train_loader):
        loss = train_batch(inputs, model, criterion, optimizer)
        # pos = (epoch + (ix+1)/_n)
        torch.save(model.state_dict(), f"epoch_{epoch}.pt")

    print("Train loss:", loss)
        
    _n = len(test_loader)
    for inputs in tqdm(test_loader):
        loss = validate_batch(inputs, model, criterion)
        # pos = (epoch + (ix+1)/_n)

    print("Valid loss:", loss)


# import cv2

# def show(img, bbs, save_path=None):
#   """
#   Shows an image with bounding boxes and optionally saves it.

#   Args:
#       img: The image as a NumPy array.
#       bbs: A list of bounding boxes as tuples (x1, y1, x2, y2).
#       save_path: The path to save the image (optional).
#   """

#   # Convert color space if needed (assuming BGR for OpenCV)
#   img = np.array(img)

#   if len(img.shape) == 3 and img.shape[2] == 3:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#   # Draw bounding boxes
#   for bb in bbs:
#     x1, y1, x2, y2 = bb
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color

#   # Save image
#   if save_path:
#     cv2.imwrite(save_path, img)
#   else:
#     # Optionally display the image (not recommended for large datasets)
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# image_paths = glob.glob(f'{DATA_ROOT}/images/*')
# # image_id = choose(test_ds.image_infos)
# image_id = "0a41fefd1f5df27f"
# img_path = os.path.join(IMAGE_ROOT, image_id) + ".jpg"
# original_image = Image.open(img_path, mode='r')
# original_image = original_image.convert('RGB')

# # for _ in range(3):
# idx = 0
# for img_path in image_paths:
# #     image_id = choose(test_ds.image_infos)
# #     img_path = os.path.join(IMAGE_ROOT, image_id) + ".jpg"
#     original_image = Image.open(img_path, mode='r')
    
#     bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5,top_k=200, device=device)
#     labels = [target2label[c.item()] for c in labels]
#     label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
#     print(bbs, label_with_conf)
#     show(original_image, bbs=bbs,  save_path=f"img{idx}.jpg")

#     if idx == 3:
#         break
#     idx += 1
    