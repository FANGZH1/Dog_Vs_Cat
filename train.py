from getdata import DogsVSCatsDataset
from torch.utils.data import DataLoader as DataLoader
from network import CNNNet
from network import MineAlexNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models

dataset_dir = './data/'             # 数据集路径
model_cp = './model/'               # 网络参数保存位置
workers = 1                         # PyTorch读取数据线程数量
batch_size = 128                    # batch_size大小(调参后128为准确率最高，64为82%，256为86%)
lr = 0.0001                         # 学习率
nepoch = 10

def train():
    datafile = DogsVSCatsDataset('train', dataset_dir)                                              # 实例化数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers)     # 用PyTorch.DataLoader封装

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    #model = CNNNet()                       # 实例化网络
    model = models.resnet18(num_classes=2)   #RESNET18
    #model = MineAlexNet()
    model = model.cuda()                # 采用GPU计算
    model.train()                       # 设定为训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # adam方法优化器
    criterion = torch.nn.CrossEntropyLoss()                         # 交叉熵损失函数

    cnt = 0             # 已训练图片数量初始化
    for epoch in range(nepoch):
        for img, label in dataloader:                       # 循环读取数据集
            img, label = Variable(img).cuda(), Variable(label).cuda()     # 将鸡蛋放在篮子中
            out = model(img)                                # 计算网络输出值，调用了网络中的forward()方法
            loss = criterion(out, label.squeeze())          # 计算损失，其中label，必须是一个1维Tensor（被坑惨了）
            loss.backward()                                 # 误差反向传播，用求导的方式，计算网络中每个节点参数的梯度
            optimizer.step()                                # 对各个参数进行调整
            optimizer.zero_grad()                           # 清除梯度
            cnt += 1
            print('Epoch:{0},Frame {1}/{2}, train_loss {3}'.format(epoch, cnt*batch_size, len(datafile),loss/batch_size))          # 打印一个批量（16）的训练结果
        cnt = 0 
    # 保存网络的参数
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))            

if __name__ == '__main__':
    train()


