import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import datetime
from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as NN

torch.cuda.set_device(0)
EVAL_DATA_PATH = Path('./data/eval')
TRAIN_DATA_PATH = Path('./data/train')
BATCH_SIZE = 64

model = torchvision.models.resnet18(pretrained = False)
model.fc = NN.Linear(512, 5)
print(model)
model = model.cuda()
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomCrop(128, padding = 4, pad_if_needed = True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation((-45, 45)), #随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
])

eval_dataset = dsets.ImageFolder(EVAL_DATA_PATH,transform = transform)
train_dataset = dsets.ImageFolder(TRAIN_DATA_PATH,transform = transform)

eval_dataloader = DataLoader(eval_dataset ,batch_size = BATCH_SIZE, shuffle=True)
train_dataloader = DataLoader(train_dataset ,batch_size = BATCH_SIZE, shuffle=True)

criterion = NN.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(0, 100):
    model.train()
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = F.nll_loss(output, target)
            correct = (predictions == target).sum().item()
            accuracy = correct / BATCH_SIZE
            
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    print("Epoch done, evaluating:", epoch)
    model.eval()
    with tqdm(eval_dataloader, unit="batch") as eepoch:
        for data, target in eepoch:
            eepoch.set_description(f"Epoch {epoch}")
            data, target = data.cuda(), target.cuda()
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == target).sum().item()
            accuracy = correct / BATCH_SIZE

            eepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "./" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".bin")

print('Finished Training')