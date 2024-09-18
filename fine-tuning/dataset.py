import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A


augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


class RDDDataset(Dataset):
    IMAGE_WIDTH = 640 
    IMAGE_HEIGHT = 640 
    # augmentation object should be passed as argument
    def __init__(self, root_dir, split='train', augment=None):
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        self.augment = augment
        self.image_paths = sorted([os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith('.jpg')])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = os.path.join(self.label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))

        image = self._load_image(image_path)
        boxes, labels = self._load_labels(label_path)

        # should not be used as global variable
        # Apply augmentations if necessary
        # if self.augment:
        #     augmented = augmentation_pipeline(image=image, bboxes=boxes, labels=labels)
        #     image = augmented['image']
        #     boxes = augmented['bboxes']

        return image, boxes, labels

    def _load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return image

    def _load_labels(self, label_path):
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                label = line.strip().split(' ')
                class_id = int(label[0])
                x_min, y_min, x_max, y_max = map(float, label[1:])
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

        return boxes, labels

    def collate_fn(self, batch):
        images = [torch.tensor(item[0]) for item in batch]
        boxes = [torch.tensor(item[1]).float() for item in batch]
        labels = [torch.tensor(item[2]).long() for item in batch]

        images = torch.stack(images, dim=0)

        return images, boxes, labels


if __name__ == "__main__":
    dataset = RDDDataset('datasets/rdd/Japan', split='train', augment=None) # aug object

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn, pin_memory=True)

    for batch_idx, inputs in enumerate(data_loader):
        img, box, lbl = inputs
        print(f"Batch {batch_idx}:")
        print(f" - Images shape: {img.shape}")
        print(box)
        print(lbl)
        break