import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, data_path):
        #data_path:数据所在文件夹
        self.data_path = data_path
        self.img_list = os.listdir(self.data_path)

    def __getitem__(self, index):
        img_title = self.img_list[index]
        img_label = int(img_title.split('_')[1].split('.')[0])
        img_path = os.path.join(self.data_path, img_title)
        img = np.load(img_path,allow_pickle=True)
        img = img.transpose(2, 0, 1)
        return img, img_label

    def __len__(self):
        return len(self.img_list)


class CCNN(nn.Module):                          ### M-CNN & C-CNN
    def __init__(self, CNN_PARAMS):
        super(CCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=2 * CNN_PARAMS['n_ch'],
                               kernel_size=(CNN_PARAMS['n_ch'], 1),
                               padding="valid")
        self.batchnorm1 = nn.BatchNorm2d(2 * CNN_PARAMS['n_ch'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(CNN_PARAMS['droprate'])

        self.conv2 = nn.Conv2d(in_channels=2 * CNN_PARAMS['n_ch'],
                               out_channels=2 * CNN_PARAMS['n_ch'],
                               kernel_size=(1, CNN_PARAMS['kernel_f']),
                               padding="valid")
        self.batchnorm2 = nn.BatchNorm2d(2 * CNN_PARAMS['n_ch'])

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2 * CNN_PARAMS['n_ch']*(CNN_PARAMS['n_fc']-CNN_PARAMS['kernel_f']+1), CNN_PARAMS['num_classes'])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class EEGNet(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate):
        super(EEGNet, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=96,
                kernel_size=(1,Samples),
                bias=False,
                padding=(0,int(Samples/2)),
            ),
            nn.BatchNorm2d(96),
            nn.Conv2d(                          #Depthwise Conv
                in_channels=96,
                out_channels=96,
                kernel_size=(Chans,1),
                groups=96,
                bias=False
            ),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=DropoutRate),
        )
        self.block2=nn.Sequential(
            nn.Conv2d(                          #Depthwise Conv
                in_channels=96,
                out_channels=96,
                kernel_size=(1,16),
                padding=(0,8),
                groups=96,
                bias=False
            ),
            nn.Conv2d(                          #Pointwise Conv
                in_channels=96,
                out_channels=96,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.ceil(math.ceil(Samples/4)/8)*96,nb_classes),
        )

    def forward(self,x):
        x=x.to(torch.float32)
        x=self.block1(x)
        x=self.block2(x)
        x=self.classification(x)
        return x



class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, dropoutRate = 0.3):
        super(DeepConvNet, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 9),padding=(0,4))
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(Chans, 1))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3),ceil_mode=True)
        self.dropout1 = nn.Dropout(dropoutRate)

        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 9),padding=(0,4))
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3),ceil_mode=True)
        self.dropout2 = nn.Dropout(dropoutRate)

        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 9),padding=(0,4))
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3),ceil_mode=True)
        self.dropout3 = nn.Dropout(dropoutRate)

        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 9),padding=(0,4))
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3),ceil_mode=True)
        self.dropout4 = nn.Dropout(dropoutRate)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(math.ceil(math.ceil(math.ceil(math.ceil(Samples/3)/3)/3)/3)*200, nb_classes)

    def forward(self, x):
        # Define the forward pass
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
