import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
import json
import numpy as np
import loss_function as loss_f
import dataset
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True
pickle_path = ""
abstract_path = ""
cityscape_path = ""
data_path = [abstract_path, cityscape_path] 
img_list = []

for i in range(2):
    for path in tqdm(glob.glob(os.path.join(data_path[i], '*.jpg'))):
        img = Image.open(path)
        if img.mode == "RGB":
            img_resize = img.resize((256, 256))
            img_np = np.array(img_resize, dtype='uint8')
            img_list.append(img_np)

with open(pickle_path, mode="wb") as f:
    pickle.dump(img_list, f)
