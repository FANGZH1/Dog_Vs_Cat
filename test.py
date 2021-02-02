#test
from getdata import DogsVSCatsDataset as DVCD
from network import CNNNet
from network import MineAlexNet
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import numpy as np
import torchvision.models as models

dataset_dir = './data/'                 # 数据集路径
model_file = './model/model.pth'        # 模型保存路径

batch_size = 128
workers = 1  

def test():
    #model = CNNNet()
    model = models.resnet18(num_classes=2)
    #model = MineAlexNet()
    model = model.cuda()  
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数 

    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout
    
    datafile = DVCD('test', dataset_dir)                # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers)

    print('test  Dataset loaded! length of train set is {0}'.format(len(datafile)))
    #cnt = np.zeros(datafile.data_size)
    #cnt = int(cnt)
    total = 0
    correct = 0
    for img,label in dataloader:                                         
        img = Variable(img).cuda()
        label = label.squeeze().cuda()                        
        y = model(img)
        _,predicted = torch.max(y.data,1)
        print(predicted) 
        print(label)
        total += label.size(0)
        correct += (predicted == label).sum()
    print(correct) 
    print(total)       
    print('Accuracy of the network on the 3000 test images: %d %%' % (100 * correct / total))                      


if __name__ == '__main__':
    test()

