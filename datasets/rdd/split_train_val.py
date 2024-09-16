import random


path = "datasets/rdd/Japan/train/Japan.txt"
train_path = "datasets/rdd/Japan/train/train.txt"
valid_path = "datasets/rdd/Japan/train/valid.txt"

with open(path, "r") as f:
    data = f.readlines()

    random.shuffle(data)

split_point = len(data) // 5
subset1 = data[:split_point]
subset2 = data[split_point:]

with open(train_path, "w") as f:
    for line in subset2:
        f.write(line)

with open(valid_path, "w") as f:
    for line in subset1:
        f.write(line)
        