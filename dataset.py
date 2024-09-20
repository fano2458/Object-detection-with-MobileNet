import torch
import glob
from PIL import Image
import os
import numpy as np
import pandas as pd
from torchvision import transforms


# DATA_ROOT = 'dataset/open-images-bus-trucks/'
DATA_ROOT = 'sample_bus_dataset/'
IMAGE_ROOT = f'{DATA_ROOT}/images'
DF_RAW = df = pd.read_csv(f'{DATA_ROOT}/sample_df.csv')

df = df[df['ImageID'].isin(df['ImageID'].unique().tolist())]

label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    w, h = 320, 320
    def __init__(self, df, image_dir):
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
        data = self.df[self.df['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin','YMin','XMax','YMax']].values
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates

        # print(type(img), type(boxes), type(labels))
        # print(img)
        # print(boxes)
        # print(labels)
        # print(img.shape) (W, H, 3)
        # np.array, list, list
        # image, [[x1, y1, x2, y2], [x1, y1, x2, y2]], ['Class_name']
        return img, boxes, labels

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for item in batch:
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/320.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
        images = torch.cat(images).to(device)
        return images, boxes, labels
    
    def __len__(self):
        return len(self.image_infos)
    



if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    trn_ids, val_ids = train_test_split(df.ImageID.unique(), test_size=0.2, random_state=99)
    # trn_ids, val_ids = train_test_split(trn_ids, train_size=0.8, random_state=99)
    trn_df, val_df = df[df['ImageID'].isin(trn_ids)], df[df['ImageID'].isin(val_ids)]
    print(len(trn_df), len(val_df))

    train_ds = OpenDataset(trn_df, IMAGE_ROOT)
    test_ds = OpenDataset(val_df, IMAGE_ROOT)

    dl = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=train_ds.collate_fn, pin_memory=False)

    for el in dl:
        img, box, lbl = el
        print(img.shape)
        print(box)
        print(lbl)

        break
