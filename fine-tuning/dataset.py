import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np


class Container:
    """
    Help class for manage boxes, labels, etc...
    Not inherit dict due to `default_collate` will change dict's subclass to dict.
    """

    def __init__(self, *args, **kwargs):
        self._data_dict = dict(*args, **kwargs)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self._data_dict[key]

    def __iter__(self):
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def _call(self, name, *args, **kwargs):
        keys = list(self._data_dict.keys())
        for key in keys:
            value = self._data_dict[key]
            if hasattr(value, name):
                self._data_dict[key] = getattr(value, name)(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        return self._call('to', *args, **kwargs)

    def numpy(self):
        return self._call('numpy')

    def resize(self, size):
        """resize boxes
        Args:
            size: (width, height)
        Returns:
            self
        """
        img_width = getattr(self, 'img_width', -1)
        img_height = getattr(self, 'img_height', -1)
        assert img_width > 0 and img_height > 0
        assert 'boxes' in self._data_dict
        boxes = self._data_dict['boxes']
        new_width, new_height = size
        boxes[:, 0::2] *= (new_width / img_width)
        boxes[:, 1::2] *= (new_height / img_height)
        return self

    def __repr__(self):
        return self._data_dict.__repr__()


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

        if self.augs:  # TODO
            pass

        label_path = self.paths[index]
        label_path = label_path.replace('images', 'labels')[:-1] + '.txt'
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            data = f.readlines()
            for line in data:
                label = line.split(sep=' ')

                class_id = int(label[0])
                x_min = float(label[1]) * 320
                y_min = float(label[2]) * 320
                x_max = float(label[3]) * 320
                y_max = float(label[4]) * 320
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )

        return image, targets, index


if __name__ == "__main__":
    sd = RDD_dataset('datasets/rdd/Japan/train/train.txt')
    print(sd[0][0].shape)
