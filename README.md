# Object-detection-with-MobileNet
This repository contains implementation of MobileNet SSD for object detection on edge devices


# TODOs 

| Done? | Tasks Description | Assignee | Reviewed? | Deadline | 
| -- | -- | -- | -- | -- |
| [x] | Re-implement [dataset.py](dataset.py). Desired inputs and outputs (inputs and shapes) can be validated using [sample_bus_dataset](sample_bus_dataset).  | @zakcination | @fano2458 | 19.09.24
| [ ] | Add learning rate scheduler, early stopping | -- | -- | 28.09.24 |
| [ ] | Add augmentations | -- | -- | 28.09.24 |
| [ ] | Add best model saving logic, graphs of metrics and losses | -- | -- | 28.09.24 |
| [ ] | Change input resolution from 640x640 to 320x320 | @fano2458 | -- | 28.09.24 |
| [ ] | Compelete training script | -- | -- | 30.09.24 |
| [ ] | Iteratively improve model performance by changing hyperparameters, model architecture (log results) | -- | -- | 12.10.24 |
| [ ] | Convert model to ONNX, use quantization, pruning, etc to improve speed of the model. (log results) | @ruslan | @fano2458 | 19.10.24
| [ ] | Write OpenVINO pipiline for inference on Python and C++. Compare performance | -- | -- | 01.11.24
| [ ] | Write iOS application and deploy model there. Measure performance | @dias | -- | 08.11.24
| [ ] | Write report and make presentation | @all | -- | 15.11.24
