# Object-detection-with-MobileNet
This repository contains implementation of MobileNet SSD for object detection on edge devices


# TODOs 
- [x] re-implement [dataset.py](dataset.py). Desired inputs and outputs (inputs and shapes) can be validated using [sample_bus_dataset](sample_bus_dataset). 
- [ ] Compelete training script (should include lr scheduler, appropriate augmentations (so that images do not look unrealistic), early stopping, graphs of metrics, losses)
- [ ] Change input resolution from 640x640 to 320x320
- [ ] Iteratively improve model performance by changing hyperparameters, model architecture (log results)
- [ ] Convert model to ONNX, use quantization, pruning, etc to improve speed of the model. (log results)
- [ ] Write OpenVINO pipiline for inference on Python and C++. Compare performance
- [ ] Write iOS application and deploy model there. Measure performance
- [ ] Write report and make presentation
