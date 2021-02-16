import numpy as np
import torch
import torch.nn as NN
import torchvision
import torchvision.transforms as transforms
import os
import argparse

import preprocessing
from PIL import Image

parser = argparse.ArgumentParser(description='Pass in an image, it will show you its class')
parser.add_argument('-i','--path', type = str, dest = 'img_path', help = 'Image file path')
parser.add_argument('-m','--model', type = str, help = 'Model file path (*.bin)', default = './chkpoint.bin')
parser.add_argument('-v','--verbose',type = bool, help = 'Show model structure and full output', default = False)
args = parser.parse_args()

torch.cuda.set_device(0) #使用 GPU

from preprocessing import transform

image = Image.open(args.img_path)
x = transform(image)
x.unsqueeze_(0)
x = x.cuda()

from model import model
model = model.cuda()
model.load_state_dict(torch.load(args.model))
model.eval()

output = model(x)

if args.verbose == True:
    print(model)
    print(output)
else:
    predictions = output.argmax(dim=1, keepdim=True).squeeze()
    print(predictions)

