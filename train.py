import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as NN
import torch.optim as optim

import torchvision
import torchvision.datasets as dsets

from pathlib import Path
import os
from tqdm import tqdm

import preprocessing

torch.cuda.set_device(0) #使用 GPU
EVAL_DATA_PATH = Path('./data/eval') #验证集路径
TRAIN_DATA_PATH = Path('./data/train') #训练集路径
BATCH_SIZE = 64 #批大小

from model import model
model = model.cuda()
print(model)

#定义训练和验证集 和 dataloader
eval_dataset = dsets.ImageFolder(EVAL_DATA_PATH,transform = preprocessing.transform)
train_dataset = dsets.ImageFolder(TRAIN_DATA_PATH,transform = preprocessing.transform)

eval_dataloader = DataLoader(eval_dataset ,batch_size = BATCH_SIZE, shuffle=True)
train_dataloader = DataLoader(train_dataset ,batch_size = BATCH_SIZE, shuffle=True)

#使用交叉熵为 loss，使用 SGD 优化方法
criterion = NN.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)


#载入上次训练
model.load_state_dict(torch.load("./chkpoint_res.bin"))

#开始训练
for epoch in range(0, 100):
    model.train()
    with tqdm(train_dataloader, unit="batch") as tepoch: #进度条
        correct = 0
        batch = 0
        for data, target in tepoch:
            batch += 1
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.cuda(), target.cuda() #数据载入 GPU

            optimizer.zero_grad() #梯度归零
            output = model(data) #前向计算
            loss = criterion(output, target) #计算 loss
            loss.backward() #反向传播
            optimizer.step() #优化器梯度下降

            predictions = output.argmax(dim=1, keepdim=True).squeeze() #预测
            correct += (predictions == target).sum().item() #统计预测正确数
            accuracy = correct / (BATCH_SIZE * batch) #计算准确度
            
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    if epoch % 15 == 0:
        print("Epoch done, evaluating:", epoch)
        torch.save(model.state_dict(), "./chkpoint_res.bin") #每 15 epoch 保存一次
        model.eval() #测试
        with tqdm(eval_dataloader, unit="batch") as eepoch:
            correct = 0
            batch = 0
            for data, target in eepoch:
                batch += 1
                eepoch.set_description(f"Epoch {epoch}")
                data, target = data.cuda(), target.cuda()
                output = model(data)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                correct += (predictions == target).sum().item()
                accuracy = correct / (BATCH_SIZE * batch)

                eepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

print('Finished Training')