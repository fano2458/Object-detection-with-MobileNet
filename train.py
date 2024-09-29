import time

import torch

from mobilenet_ssd.ssdlite import SSDLite

global device 

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def train(train_loader, model, criterion, optimizer, epoch, grad_clip):
    print_freq = 200
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i , (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)
        
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:                   
            clip_gradient(optimizer, grad_clip)

        optimizer.step()
        
        #print(loss.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        #print('%d done...' %i)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            # logger.debug('Epoch: [{0}][{1}/{2}]\t'
            #              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
            #              'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
            #                                                       batch_time=batch_time,
            #                                                       data_time=data_time, loss=losses))
            

def validate(val_loader, model, criterion):
    print_freq = 200
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_scores = model(images)
            
            predicted_locs = predicted_locs.to(device)
            predicted_scores = predicted_scores.to(device)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
            #     logger.debug('[{0}/{1}]\t'
            #                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
            #                                                           batch_time=batch_time,
            #                                                           loss=losses))
            
            # logger.debug('\n * Loss - {loss.avg:.3f}\n'.format(loss=losses))

            return losses.avg



def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SSDLite(class_num=2, backbone='MobileNetV3_Large', device=device).to(device)

    # datasets 
    

if __name__ == "__main__":
    main()
