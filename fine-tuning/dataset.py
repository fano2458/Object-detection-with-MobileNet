import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Constants for image dimensions (adjustable for experimentation)
IMAGE_WIDTH = 640  # Change as needed
IMAGE_HEIGHT = 640  # Change as needed

# Albumentations augmentation pipeline (modify augmentations as needed)
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


class RDDDataset(Dataset):
    """
    PyTorch Dataset for RDD object detection dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split to use ('train', 'val', or 'test').
        augment (bool): Whether to apply augmentations.
    """

    def __init__(self, root_dir, split='train', augment=False):
        """
        Initialize the dataset by loading image and label paths.

        Args:
            root_dir (str): The root directory where the dataset is stored.
            split (str): Dataset split to use ('train', 'val', or 'test'). Default is 'train'.
            augment (bool): Whether to apply augmentations. Default is False.
        """
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        self.augment = augment
        self.image_paths = sorted([os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith('.jpg')])

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Get the image and corresponding labels at a specific index.

        Args:
            index (int): Index of the image-label pair to retrieve.

        Returns:
            tuple: (image, targets), where image is the augmented or original image tensor,
            and targets is a dictionary holding bounding boxes and labels.
        """
        image_path = self.image_paths[index]
        label_path = os.path.join(self.label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))

        # Load image and labels
        image = self._load_image(image_path)
        boxes, labels = self._load_labels(label_path)

        # Apply augmentations if necessary
        if self.augment:
            augmented = augmentation_pipeline(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']

        # Convert boxes and labels into tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Return images in tensor format (C, H, W)
        return torch.tensor(image).permute(2, 0, 1), {'boxes': boxes, 'labels': labels}

    def _load_image(self, image_path):
        """Load the image from a given path."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return image

    def _load_labels(self, label_path):
        """
        Load and preprocess labels from a label file in YOLO format (x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized).
        """
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                label = line.strip().split(' ')
                class_id = int(label[0])
                x_min_norm, y_min_norm, x_max_norm, y_max_norm = map(float, label[1:])

                # Convert normalized coordinates (YOLO format) to absolute pixel values
                x_min = x_min_norm * IMAGE_WIDTH
                y_min = y_min_norm * IMAGE_HEIGHT
                x_max = x_max_norm * IMAGE_WIDTH
                y_max = y_max_norm * IMAGE_HEIGHT

                boxes.append([x_min / IMAGE_WIDTH, y_min / IMAGE_HEIGHT, (x_max - x_min) / IMAGE_WIDTH, (y_max - y_min) / IMAGE_HEIGHT])
                labels.append(class_id)

        return boxes, labels


def collate_fn(batch):
    """
    Custom collate function to handle batches with varying number of bounding boxes.
    
    Args:
        batch (list): List of tuples (image, targets) from the dataset.
    
    Returns:
        tuple: Batch of images and target dictionaries containing padded bounding boxes and labels.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a tensor (batch size, channels, height, width)
    images = torch.stack(images, dim=0)

    # Collate boxes and labels
    all_boxes = [t['boxes'] for t in targets]
    all_labels = [t['labels'] for t in targets]

    return images, {'boxes': all_boxes, 'labels': all_labels}


if __name__ == "__main__":
    dataset = RDDDataset('datasets/rdd/Japan', split='train', augment=True)

    # Create DataLoader with custom collate_fn and efficient parameters
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    for batch_idx, (images, targets) in enumerate(data_loader):
        print(f"Batch {batch_idx}:")
        print(f" - Images shape: {images.shape}")
        print(f" - Number of bounding boxes: {[len(boxes) for boxes in targets['boxes']]}")
        break