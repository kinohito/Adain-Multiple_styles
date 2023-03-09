import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
from torchvision.utils import save_image

import loss_function as loss_f
import dataset
import model

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__  == "__main__":
    style_path = ""
    content_path = ""
    #データセット定義
    dataset_train = dataset.PreprocessDataset(content_path, style_path)

    #検証、訓練分割
    batch_size = 8

    #データローダー定義
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    lr = 1e-5
    c_weight = 1.0
    s_weight = 10.0
    n_epochs = 10
    alpha = 1
    model_path = ""

    net = model.Style_transfer(alpha).to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    net.load_state_dict(torch.load(model_path))
    filepath = ""
 
    net.train()
    n_train = 0

    for k in range(n_epochs):
        for i, (content, style) in enumerate(dataloader_train, 1):
            batch_wiki = style
            batch_places = content

            batch_wiki = batch_wiki.to(device)  # テンソルをGPUに移動
            batch_places = batch_places.to(device)  # テンソルをGPUに移動

            kansei_image, s_loss, c_loss = net.forward(batch_places, batch_wiki)  # 順伝播

            loss = c_weight * c_loss + s_weight * s_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("count" + str(i) + " : "  + str(loss))

            if i % 500 == 0:
                name = "Iter" + str(i) +"_epoch" + str(k) + ".pth"
                model_path = filepath + name
                torch.save(net.state_dict(), model_path)
            
            if i % 500 == 0:
                batch_wiki = dataset.denorm(batch_wiki, device)
                batch_places = dataset.denorm(batch_places, device)
                kansei_image = dataset.denorm(kansei_image, device)
                input_list = torch.cat([batch_places, batch_wiki, kansei_image], dim=0)
                input_list = input_list.to('cpu')
                save_image(input_list, f'{filepath}/{k}_epoch_{i}_iteration.png', nrow=batch_size)






    
    
    
