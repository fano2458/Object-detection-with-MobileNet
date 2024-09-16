import os
import tqdm
from glob import glob


path = '/home/fano/Desktop/CSCI 494/final_project/Object-detection-with-MobileNet/datasets/rdd/Japan/train/'

files = glob(path+'images/*')

print(len(files))
with open(path+'Japan.txt', 'w') as f:
    for file in tqdm.tqdm(files):
        # print(file)
        f.write(file[74:-4] + '\n')
