import pandas as pd
import torch

import collections, os, torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy as np
import os
import glob
from tqdm import tqdm


DATA_ROOT = 'dataset/open-images-bus-trucks/'
IMAGE_ROOT = f'{DATA_ROOT}/images'
DF_RAW = df = pd.read_csv(f'{DATA_ROOT}/df.csv')

df = df[df['ImageID'].isin(df['ImageID'].unique().tolist())]

label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print(num_classes)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()
    
class OpenDataset(torch.utils.data.Dataset):
    w, h = 640, 640
    def __init__(self, df, image_dir=IMAGE_ROOT):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir+'/*')
        self.df = df
        self.image_infos = df.ImageID.unique()
        
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path =  os.path.join(self.image_dir, image_id) + ".jpg"
        # img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        data = df[df['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin','YMin','XMax','YMax']].values
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates
        return img, boxes, labels

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for item in batch:
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/640.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
        images = torch.cat(images).to(device)
        return images, boxes, labels
    def __len__(self):
        return len(self.image_infos)
    

trn_ids, val_ids = train_test_split(df.ImageID.unique(), test_size=0.1, random_state=99)
trn_df, val_df = df[df['ImageID'].isin(trn_ids)], df[df['ImageID'].isin(val_ids)]
len(trn_df), len(val_df)


train_ds = OpenDataset(trn_df)
test_ds = OpenDataset(val_df)


train_loader = DataLoader(train_ds, batch_size=32, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, collate_fn=test_ds.collate_fn, drop_last=True)

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


from src.mobilenet_ssd import MultiBoxLoss, MobileNetSSD
from src.detect import *

n_epochs = 3

model = MobileNetSSD(num_classes).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

for epoch in range(n_epochs):
    _n = len(train_loader)
    for inputs in tqdm(train_loader):
        loss = train_batch(inputs, model, criterion, optimizer)
        # pos = (epoch + (ix+1)/_n)

    print("Train loss:", loss)
        
    _n = len(test_loader)
    for inputs in tqdm(test_loader):
        loss = validate_batch(inputs, model, criterion)
        # pos = (epoch + (ix+1)/_n)

    print("Valid loss:", loss)


import cv2

def show(img, bbs, save_path=None):
  """
  Shows an image with bounding boxes and optionally saves it.

  Args:
      img: The image as a NumPy array.
      bbs: A list of bounding boxes as tuples (x1, y1, x2, y2).
      save_path: The path to save the image (optional).
  """

  # Convert color space if needed (assuming BGR for OpenCV)
  img = np.array(img)

  if len(img.shape) == 3 and img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Draw bounding boxes
  for bb in bbs:
    x1, y1, x2, y2 = bb
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color

  # Save image
  if save_path:
    cv2.imwrite(save_path, img)
  else:
    # Optionally display the image (not recommended for large datasets)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_paths = glob.glob(f'{DATA_ROOT}/images/*')
# image_id = choose(test_ds.image_infos)
image_id = "0a41fefd1f5df27f"
img_path = os.path.join(IMAGE_ROOT, image_id) + ".jpg"
original_image = Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

# for _ in range(3):
idx = 0
for img_path in image_paths:
#     image_id = choose(test_ds.image_infos)
#     img_path = os.path.join(IMAGE_ROOT, image_id) + ".jpg"
    original_image = Image.open(img_path, mode='r')
    
    bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    print(bbs, label_with_conf)
    show(original_image, bbs=bbs,  save_path=f"img{idx}.jpg")

    if idx == 3:
        break
    idx += 1
    