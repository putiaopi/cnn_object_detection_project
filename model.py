import torchvision
import torch.nn as NN

#model = torchvision.models.alexnet(pretrained=False) #使用 alexnet
model = torchvision.models.resnet34(pretrained=False) #使用 resnet34
#model.fc = NN.Linear(4096, 5) #修改最后一层线性激活层为 4096->5，以匹配数据集(alex)
model.fc = NN.Linear(512, 5)   #修改最后一层线性激活层为 512->5，以匹配数据集(res)
