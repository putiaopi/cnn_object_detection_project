import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os

import torch.nn.functional as F
import torch.nn as NN

torch.cuda.set_device(0)
EVAL_DATA_PATH = Path('./data/eval')
TRAIN_DATA_PATH = Path('./data/train')
BATCH_SIZE = 64

model = torchvision.models.resnet18(pretrained = False)
model = model.cuda()
transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4, pad_if_needed = True),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation((-45, 45)), #随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
])

eval_dataset = dsets.ImageFolder(EVAL_DATA_PATH,transform = transform)
train_dataset = dsets.ImageFolder(TRAIN_DATA_PATH,transform = transform)

eval_dataloader = DataLoader(eval_dataset ,batch_size = BATCH_SIZE)
train_dataloader = DataLoader(train_dataset ,batch_size = BATCH_SIZE)

criterion = NN.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(10):  # 多批次循环

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # 获取输入
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            torch.save(model.state_dict(), ".")

print('Finished Training')