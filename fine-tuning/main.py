import torch
import math
from model import get_model
from dataset import RDD_dataset
from torch.utils.data import DataLoader


def adjust_learning_rate(optimizer, epoch, warmup=False, warmup_ep=0, enable_cos=True):
    lr = 3e-4
    if warmup and epoch < warmup_ep:
        lr = lr / (warmup_ep - epoch)
    elif enable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_ep) / (100 - warmup_ep)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, optimizer, train_dataloader, epoch):
    pass


def evaluate(model, valid_dataloader, epoch):
    pass


def main():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'

    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-5)

    train_ds = RDD_dataset("datasets/rdd/Japan/train/train.txt")
    valid_ds = RDD_dataset("datasets/rdd/Japan/train/valid.txt")
    
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=False)
    valid_dataloader = DataLoader(valid_ds, batch_size=64, shuffle=False, pin_memory=False)

    for epoch in range(100):
        adjust_learning_rate(optimizer, epoch)
        train(model, optimizer, train_dataloader, epoch) # assuming loss is returned by model it self
        evaluate(model, valid_dataloader, epoch)

        # TODO implement early stop based on validation loss

if __name__ == "__main__":
    main()