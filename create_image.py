import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
import random

import json
import base64
import io

import loss_function as loss_f
import dataset
import model

#計算デバイス定義
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#jsonファイル読み込み
json_file = input("jsonファイルのパスを入力してください : ")
f = open(json_file)
json_data = json.load(f)
label_num = len(json_data["shapes"]) + 1

#画像読み込み
style_path_list = []
style_image_list = []
content_path = input("コンテンツ画像のパスを入力してください : ")
content_image = Image.open(content_path)

style_path_list.append(input("スタイル画像のパスを入力してください(背景) : "))
style_image_list.append(Image.open(style_path_list[0]))

for num in range(label_num-1):
    style_path_list.append(input("スタイル画像のパスを入力してください(" + json_data["shapes"][num]["label"] +") : "))
    style_image_list.append(Image.open(style_path_list[num+1]))

points_list = []
# アノテーションの頂点座標を取得
for i in range(label_num-1):
    points = json_data['shapes'][i]['points'] 
    # list -> tuple
    points_list.append([tuple(point) for point in points]) # or list(map(tuple, points))

# マスク画像を生成
w, h = content_image.size
mask = Image.new('L', (w, h))
draw = ImageDraw.Draw(mask)
for i in range(label_num-1):
    draw.polygon(points_list[i], fill = (i+1))

plt.imshow(mask)
plt.show()

#コンテンツ画像変換
trans_content = dataset.ImageTransform_test_content()
trans_mask = dataset.ImageTransform_mask()
content_tensor = trans_content(content_image).unsqueeze(0).to(device)
mask_numpy = np.array(trans_mask(mask))

content_size = content_tensor.shape[2:4]

#スタイル画像変換
trans_style = dataset.ImageTransform_test_style(content_tensor.shape[2:])
style_tensor_list = []
for num in range(label_num):
    style_tensor_list.append(trans_style(style_image_list[num]).unsqueeze(0).to(device))

trans_resize = dataset.ImageTransform_resize(content_tensor.shape[2:])

#モデル読み込み
alpha = 0
net_list = []
model_path = ''

alpha = float(input("αを入力してください。スタイル画像(α) : コンテンツ画像(1-α) 背景 : "))
net = model.Style_transfer(alpha).to(device)
net.load_state_dict(torch.load(model_path))
net_list.append(net)

for num1 in range(label_num-1):
    alpha = float(input("αを入力してください。スタイル画像(α) : コンテンツ画像(1-α) " + json_data["shapes"][num1]["label"] +" : "))
    net = model.Style_transfer(alpha).to(device)
    net.load_state_dict(torch.load(model_path))
    net_list.append(net)

output_image_list = []

trans_resize = transforms.Resize(content_size)

#変換
for i in range(label_num):
    with torch.no_grad():
        image, _, _ = net_list[i].forward(content_tensor, style_tensor_list[i])
        output_image_list.append(trans_resize(image.view(image.shape[1:4])))

#画素値の置き換え
output_numpy = output_image_list[0].to('cpu').detach().numpy().copy()

for i in range(label_num-1):
    index = mask_numpy == (i+1)
    index = np.repeat(index[None, :], 3, axis=0)
    np.place(output_numpy, index, (output_image_list[i+1].to('cpu').detach().numpy().copy())[index])

batch_wiki_list = []
kansei_image_list = []

#コンテンツ画像
kansei_image_list.append(dataset.denorm(content_tensor, device))

#正規化を戻す:スタイル画像
for i in range(label_num):
    kansei_image_list.append(dataset.denorm(style_tensor_list[i], device))

#出力画像
kansei_image  = torch.from_numpy(output_numpy.astype(np.float32)).clone().to(device)
kansei_image = dataset.denorm(kansei_image, device).view(1, 3, content_size[0], content_size[1])
kansei_image_list.append(kansei_image)

for i in range(len(kansei_image_list)):
    kansei_image_list[i] = kansei_image_list[i][0]

output = torch.stack(kansei_image_list, dim=0)
output = output.to('cpu')

filename_end = input("出力ファイル名を入力してください : ")

filename = "" + str(filename_end) + ".png"
torchvision.utils.save_image(output, filename)
