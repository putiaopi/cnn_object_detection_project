# README
## 关于
此项目意在使用 CNN 进行物体识别，数据集为危险物品的 X 光透射照片，如 手枪，剪刀，扳手等。如下所示

![1.6](https://github.com/neoncloud/cnn_object_detection_project/raw/main/media/16118317390713/1.6.jpg)
![3.29](https://github.com/neoncloud/cnn_object_detection_project/raw/main/media/16118317390713/3.29.jpg)



## 进度与方法
### train.py
目前已经使用 torch 自带的 ResNet34 和 AlexNet 训练出了两个模型（chkpoint.bin），效果拔群。

运行 train.py 可以开始训练，若不提供 chkpoint.bin（不继续训练）则需要注释掉第 88 行。环境依赖为torch通用环境。

数据集放在 ./data 下面，目录结构如下

```
./data
├── eval
│   ├── 0
│   │   ├── 0.1007.jpg
│   │   ├── 0.1008.jpg
│   │   ├── 0.1016.jpg
│   │   ├── 0.1017.jpg
│   │   ├── 0.1024.jpg
│   │   ├── 0.1025.jpg
│   ├── 1
│   │   ├── 1.1001.jpg
│   │   ├── 1.1002.jpg
│   │   ├── 1.1011.jpg
│   │   ├── 1.1012.jpg
│   ├── 2
│   │   ├── 2.1001.jpg
│   │   ├── 2.1002.jpg
│   │   ├── 2.1011.jpg
│   │   ├── 2.1012.jpg
...
├── train
│   ├── 0
│   │   ├── 0.1007.jpg
│   │   ├── 0.1008.jpg
│   │   ├── 0.1016.jpg
│   │   ├── 0.1017.jpg
│   │   ├── 0.1024.jpg
│   │   ├── 0.1025.jpg
│   ├── 1
│   │   ├── 1.1001.jpg
│   │   ├── 1.1002.jpg
│   │   ├── 1.1011.jpg
│   │   ├── 1.1012.jpg
│   ├── 2
│   │   ├── 2.1001.jpg
│   │   ├── 2.1002.jpg
│   │   ├── 2.1011.jpg
│   │   ├── 2.1012.jpg
...


```

### eval.py
目前已经实现验证器，向程序传入图片路径和模型参数（chkpoint.bin）可以得到所训练模型的对于此图片的分类的预测。
```
usage: eval.py [-h] [-i IMG_PATH] [-m MODEL] [-v VERBOSE]

Pass in an image, it will show you its class

optional arguments:
  -h, --help            show this help message and exit
  -i IMG_PATH, --path IMG_PATH
                        Image file path
  -m MODEL, --model MODEL
                        Model file path (*.bin)
  -v VERBOSE, --verbose VERBOSE
                        Show model structure and full output
```
使用例：
```
python3 eval.py -i '/home/neoncloud/project/data/eval/3/3.3.jpg' -m '/home/neoncloud/project/chkpoint_res.bin'
output: tensor(3, device='cuda:0')
```
可见模型成功地对图片标签进行了预测，输入为 3 号类的图片，预测为 3。

### ft_train.py
此脚本试图用预训练脚本进行 fine tuning。以 ResNet34 为例，其网络结构如下：
```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (4): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (5): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=5, bias=True)
)
```

宏观上，网络由
* layer1
* layer2
* layer3
* layer4
* avgpool
* fc

五层网络构成，我们只训练 layer4、avgpool 和 fc 层，而固定前面的网络参数不动，在程序中设置了

```
for param in model.parameters(): #先锁定网络的所有参数，后面来决定要训练的部分网络
    param.requires_grad = False

from itertools import chain #把后半截（layer4,avgpool,fc）的网络参数启动训练
for param in chain(model.layer4.parameters(), model.avgpool.parameters(), model.fc.parameters()):
    param.requires_grad = True
```

以单独训练后部网络参数。预训练参数于 torch 官网下载。

对网络进行 fine tuning 能极大节省训练时间：程序仅仅在 45 个 epoch 后正确率便达到了 91.9%
```
Epoch done, evaluating: 45
Epoch 45: 100%|████████████████████████████████████████████████████████████████| 53/53 [00:08<00:00,  6.25batch/s, accuracy=91.9, loss=0.0289]
```
作为对比，从 0 开始训练至少需要 80 个 epoch 才能达到相同水平。


## TODO
* [x] 基本可用的训练脚本，包括数据集对象（继承自 ImageFolder 类），数据集加载器对象（继承自 dataloader），简单的数据集增强（Transform，包含 padding 为正方形，随机旋转等），使用 SGD 和 交叉熵，以及一个装逼的进度条。
* [x] 设计一个友好的验证脚本（eval.py），从命令行中传入测试一个图片，并通过网络预测其标签。
* [x] 改进训练脚本，设计命令行参数，分离出 config.py（超参数设置），和 network.py (网络定义部分)。
* [x] 进行 fine tuning，测试其效果。
* [ ] 改进数据增强，提高模型的正确率。
* [ ] 自己设计一个网络并训练它，并希望还能有一个不错的正确率。
