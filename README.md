# Object-detection-with-MobileNet
This repository contains implementation of MobileNet SSD for object detection on edge devices


# TODOs 
* reimplement [main.py](main.py) (Dataset class + collate_fn method) to work with rdd dataset, Dataset class should be simillar to one in [main.py](fine-tuning/main.py). Desired inputs and outputs (inputs and shapes) can be validated using [sample_bus_dataset](sample_bus_dataset).