import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np


class RDD_dataset(Dataset):
    def __init__(self, path, augmentations=None):
        super().__init__()
        self.augs = augmentations

        with open(path, 'r') as f:
            self.paths = f.readlines()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        image_path = image_path[:-1] + '.jpg'
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (320, 320))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0

        if self.augs: # TODO 
            pass

        label_path = self.paths[index]
        label_path = label_path.replace('images', 'labels')[:-1] + '.txt'
        image_labels = []
        with open(label_path, 'r') as f:
            data = f.readlines()
            for line in data:
                label = line.split(sep=' ')

                class_id = int(label[0])
                x_min = float(label[1]) * 320
                y_min = float(label[2]) * 320
                x_max = float(label[3]) * 320
                y_max = float(label[4]) * 320

                image_labels.append([class_id, x_min, y_min, x_max, y_max])            

        return image, image_labels


if __name__ == "__main__":
    sd = RDD_dataset('datasets/rdd/Japan/train/train.txt')
    print(sd[0][0].shape)
