import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import loss_function as loss_f
from torchvision.models import vgg19

path = "/content/drive/MyDrive/3D/adain/vgg19-dcbb9e9d.pth"

class Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        self.define_module()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
    
    def define_module(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),         
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),     
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),     
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),     
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def make_vgg():
    net = Vgg19()
    load_weights = torch.load(path)
    net.load_state_dict(load_weights)
    list_module = list(net.children())

    model_relu1_1 = nn.Sequential(list_module[0][:2])
    model_relu2_1 = nn.Sequential(list_module[0][2:7])
    model_relu3_1 = nn.Sequential(list_module[0][7:12])
    model_relu4_1 = nn.Sequential(list_module[0][12:21])

    return model_relu1_1, model_relu2_1, model_relu3_1, model_relu4_1

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4
 
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.define_module()

    def define_module(self):
        self.decoder = nn.Sequential(
            #32×32
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            #64×64
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            #128×128
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            #256×256
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

def calc_mean_std(feature):
    batch_size, c = feature.size()[:2]
    features_mean = feature.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = feature.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_std, features_mean

def adain(content_features, style_features):
    content_std, content_mean = calc_mean_std(content_features)
    style_std, style_mean = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

class Style_transfer(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.enc = VGGEncoder()
        self.decoder = Decoder()

    def forward(self, c_in, s_in):
        c_feature = self.enc(c_in, True)
        s_feature = self.enc(s_in, True)

        a_feature = adain(c_feature, s_feature)

        t = (1 - self.alpha) * c_feature + self.alpha * a_feature

        image = self.decoder(t)

        image_feature1, image_feature2, image_feature3, image_feature4 = self.enc(image)
        s_feature1, s_feature2, s_feature3, s_feature4 = self.enc(s_in)

        s_loss1 = loss_f.style_loss(s_feature1, image_feature1)
        s_loss2 = loss_f.style_loss(s_feature2, image_feature2)
        s_loss3 = loss_f.style_loss(s_feature3, image_feature3)
        s_loss4 = loss_f.style_loss(s_feature4, image_feature4)

        s_total_loss = s_loss1 + s_loss2 + s_loss3 + s_loss4

        c_loss = loss_f.content_loss(t, image_feature4)

        return image, s_total_loss, c_loss

