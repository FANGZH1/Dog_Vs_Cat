# getdata.py
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# 默认图片大小
IMAGE_H = 200
IMAGE_W = 200

# 将图像数据转换成Tensor形式
data_transform = transforms.Compose([
    transforms.ToTensor()   #默认归一化到[0.0, 1.0]
])

class DogsVSCatsDataset(data.Dataset):      # 新建一个DogsVSCatsDataset类，继承Torch.data.Dataset
    def __init__(self, mode, dir):          # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.mode = mode
        self.list_img = []                  # 存放图片的路径
        self.list_label = []          # 存放图片为猫(0)或狗(1)的标签
        self.data_size = 0                  # 数据集大小
        self.transform = data_transform     # 转换关系

        if self.mode == 'train':            # 训练模式
            dir = dir + '/train/' 
            for file in os.listdir(dir): 
                self.list_img.append(dir + file)        # 将图片路径和文件名添加至image list
                self.data_size += 1
                name = file.split(sep='.')              # 分割文件名
                if name[0] == 'cat':
                    self.list_label.append(0)         # 图片为猫，label为0
                else:
                    self.list_label.append(1)         # 图片为狗，label为1
        elif self.mode == 'test':           # 测试集模式
            dir = dir + '/test/'            # 测试集路径
            for file in os.listdir(dir):
                self.list_img.append(dir + file)    # 添加图片路径至image list
                self.data_size += 1
                name = file.split(sep='.')              # 分割文件名
                if name[0] == 'cat':
                    self.list_label.append(0)         # 图片为猫，label为0
                else:
                    self.list_label.append(1)         # 图片为狗，label为1
        else:
            return print('Undefined Dataset!')

    def __getitem__(self, item):            # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':                                        # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])                       # 打开图片
            img = img.resize((IMAGE_H, IMAGE_W))                        # 图片统一成默认大小
            img = np.array(img)[:, :, :3]                               # 数据转换成numpy数组形式
            label = self.list_label[item]                               # 获取image对应的label
            return self.transform(img), torch.LongTensor([label])       # 将image和label转换成Tensor形式并返回
        elif self.mode == 'test':                                       # 测试集只读取image
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))
            img = np.array(img)[:, :, :3]
            label = self.list_label[item]
            return self.transform(img),torch.LongTensor([label])                                  # 只将image转换成Tensor形式并返回
        else:
            print('ERROR')
    def __len__(self):
        return self.data_size               # 返回数据集大小
