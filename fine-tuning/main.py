import torch
from torch.utils.data import DataLoader

from model import SSDDetector
from dataset import RDD_dataset
# from loss import MultiBoxLoss

import math
import numpy as np
from tqdm import tqdm


device = None

from torch.utils.data.dataloader import default_collate
from dataset import Container

class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids


def adjust_learning_rate(optimizer, epoch, warmup=False, warmup_ep=0, enable_cos=True):
    lr = 3e-4
    if warmup and epoch < warmup_ep:
        lr = lr / (warmup_ep - epoch)
    elif enable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_ep) / (100 - warmup_ep)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, optimizer, train_dataloader, epoch):
    model.train()
    global device

    for data in tqdm(train_dataloader, desc="Training"):
        # print(data)
        imgs, targets, _ = data
        imgs, targets = imgs.to(device), targets.to(device)
 
        # print(targets)

        outputs = model(imgs, targets=targets)

        # print(type(outputs))

        break


@torch.no_grad
def evaluate(model, valid_dataloader, epoch):
    model.eval()
    global device

    for data in tqdm(valid_dataloader, desc="Evaluation"):
        print(data)
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        for i in outputs:       # for each image
            pred_scores = outputs[i]['scores'].detach().cpu().numpy()
            boxes = outputs[i]['boxes'].detach().cpu().numpy().astype(np.int32)
            labels = outputs[i]['labels'][:len(boxes)]

            print(pred_scores.shape)       # 300 scores
            print(boxes.shape)             # (300, 4) box points
            print(labels.shape)            # 300 labels

        break


def main():
    global device
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'

    model = SSDDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-5)

    train_ds = RDD_dataset("datasets/rdd/Japan/train/train.txt")
    valid_ds = RDD_dataset("datasets/rdd/Japan/train/valid.txt")
    
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, pin_memory=False, collate_fn=BatchCollator(True))
    valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=False, pin_memory=False, collate_fn=BatchCollator(False))

    # loss_fn = MultiBoxLoss(3) # TODO

    for epoch in range(100): # TODO
        adjust_learning_rate(optimizer, epoch)
        train(model, optimizer, train_dataloader, epoch)
        # evaluate(model, valid_dataloader, epoch)
        break
        # TODO implement early stop based on validation loss


if __name__ == "__main__":
    main()
