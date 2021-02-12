import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as NN
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import datetime
from tqdm import tqdm
import numbers

torch.cuda.set_device(0)
EVAL_DATA_PATH = Path('./data/eval')
TRAIN_DATA_PATH = Path('./data/train')
BATCH_SIZE = 64

model = torchvision.models.alexnet(pretrained=False)
model.fc = NN.Linear(4096, 5)
print(model)
model = model.cuda()

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)



transform = transforms.Compose([
#    transforms.ColorJitter(brightness = 0.5),
    transforms.RandomGrayscale(p=0.9),
    NewPad(padding_mode = "symmetric"),
    transforms.Resize(size = (128,128)),
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
            loss = criterion(output, target)
            correct = (predictions == target).sum().item()
            accuracy = correct / BATCH_SIZE
            
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    print("Epoch done, evaluating:", epoch)

    if epoch % 15 == 0:
        torch.save(model.state_dict(), "./" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".bin")
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

print('Finished Training')