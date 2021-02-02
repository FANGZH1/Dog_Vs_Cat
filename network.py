import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as functional


class CNNNet(nn.Module):                                       # 新建一个网络类，继承Torch.nn.Module父类
    def __init__(self):                                     # 构造函数，设定网络层
        super(CNNNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # 第一卷积层，输入通道3，输出通道16，核大小3×3，padding大小1
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 第二卷积层，输入通道16，输出通道16，核大小3×3，padding大小1

        self.fc1 = nn.Linear(50*50*16, 128)                 # 第一线性全连接层，输入数50×50×16，输出数128
        self.fc2 = nn.Linear(128, 64)                       # 第二线性全连接层，输入数128，输出数64
        self.fc3 = nn.Linear(64, 2)                         # 第三线性全连接层，输入数64，输出数2

    def forward(self, x):                            # 重写forward方法
        x = self.conv1(x)                            # 第一次卷积
        x = functional.relu(x)                       # ReLU激活
        x = functional.max_pool2d(x, 2)              # 第一次最大池化，2×2

        x = self.conv2(x)                            # 第二次卷积
        x = functional.relu(x)                       # ReLU激活
        x = functional.max_pool2d(x, 2)              # 第二次最大池化，2×2

        x = x.view(x.size()[0], -1)                  # 将输入的[50×50×16]格式数据排列成[40000×1]形式
        x = functional.relu(self.fc1(x))             # 第一次全连接，ReLU激活
        x = functional.relu(self.fc2(x))             # 第二次全连接，ReLU激活
        y = self.fc3(x)                              # 第三次激活

        return y        

# AlexNet网络结构
class MineAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MineAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        y = self.logsoftmax(x)
        return y

