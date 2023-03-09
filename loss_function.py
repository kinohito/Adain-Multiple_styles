import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import model

def style_loss(s_feature, image_feature):
    s_std, s_mean = model.calc_mean_std(s_feature)
    image_std, image_mean = model.calc_mean_std(image_feature)

    mean_loss = F.mse_loss(image_mean, s_mean)
    std_loss = F.mse_loss(image_std, s_std)
    total_loss = mean_loss + std_loss
    
    return total_loss

def content_loss(adain_feature, image_feature):
    c_loss = F.mse_loss(image_feature, adain_feature)
    return c_loss
