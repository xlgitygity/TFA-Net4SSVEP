import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader,Dataset
# from torchsummary import summary
# from torchmetrics import Accuracy
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
# import multiprocessing as mp
# from time import time
import random
import math


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class InceptionBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


class FrequencyAttentionModul(nn.Module):
    def __init__(self):
        super(FrequencyAttentionModul, self).__init__()

    def forward(self, x):
        _, _, _, width = x.size()
        # 创建一个大小为[1, width]的平均池化层
        avgpool = nn.AvgPool2d((1, width))
        # 对输入特征图进行池化
        FrequencyWeight = avgpool(x)
        x = FrequencyWeight * x

        return x


class ChannelRecombinationModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, out_channel):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelRecombinationModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, out_channel),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, out_channel),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
        )

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x1=self.pointwise_conv(x)

        # 乘积获得结果
        x2 = Mc * x1

        return x2



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
        img = img.transpose(2,0,1)
        return img, img_label
        # return img, img_label, img_title

    def __len__(self):
        return len(self.img_list)




class MyNet(nn.Module):                                                      #毕设CNN-SWIFT
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(MyNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,15),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,15),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(
                in_channels=2 * N1,
                out_channels=2 * N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=2 * N1,
                out_channels=2 * N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(2*N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv=nn.Sequential(
            nn.Conv2d(
            in_channels=2*N1,
            out_channels=2*N1,
            kernel_size=(3,15),
            groups=2*N1,
            bias=False,
            padding=(1,7)
            ),
            nn.BatchNorm2d(2*N1),
            nn.ELU(),
        )
        self.cbam=CBAM(2*N1)
        self.conv4=nn.Sequential(
            nn.Conv2d(
                in_channels=2 * N1,
                out_channels=N2,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv=nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(3,15),
                groups=N2,
                bias=False,
                padding=(1,7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,5),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/15)/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    def forward(self,x,labels=None):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.conv1(x_l)
        x_h=self.conv2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x2=self.conv3(x1)
        x3=self.depthwise_conv(x2)
        x4=self.cbam(x3)
        x5=self.conv4(x4)
        x6=self.separable_conv(x5)
        x7=self.classification(x6)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x7,labels)
            return x7,loss
        else:
            return x7



class FrequencyAttention_ChannelRecombination_AdaptivePool_Net(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(FrequencyAttention_ChannelRecombination_AdaptivePool_Net, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.FA_module=FrequencyAttentionModul()
        self.CR_module=ChannelRecombinationModul(2*N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None,visual=False):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_fa=self.FA_module(x1)
        x_cr=self.CR_module(x_fa)
        x2=self.conv1(x_cr)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6
        # if labels is not None:
        #     loss = loss_function(x6, labels)
        #     if visual:
        #         return x6, loss,x1, x_fa, x_cr , x2, x3, x4
        #     else:
        #         return x6, loss
        # else:
        #     if visual:
        #         return x6,x1, x_fa, x_cr , x2, x3, x4
        #     else:
        #         return x6


class TFANet(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(TFANet, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.FA_module=FrequencyAttentionModul()
        self.CR_module=ChannelRecombinationModul(2*N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None,visual=False):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_fa=self.FA_module(x1)
        x_cr=self.CR_module(x_fa)
        x2=self.conv1(x_cr)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x_cr,loss
        else:
            return x_cr

class TFA_Net_Pool_Dropout(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(TFA_Net_Pool_Dropout, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.FA_module=FrequencyAttentionModul()
        self.CR_module=ChannelRecombinationModul(2*N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None,visual=False):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_fa=self.FA_module(x1)
        x_cr=self.CR_module(x_fa)
        x2=self.conv1(x_cr)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x_cr,loss
        else:
            return x_cr
        # if labels is not None:
        #     loss = loss_function(x6, labels)
        #     if visual:
        #         return x6, loss,x1, x_fa, x_cr , x2, x3, x4
        #     else:
        #         return x6, loss
        # else:
        #     if visual:
        #         return x6,x1, x_fa, x_cr , x2, x3, x4
        #     else:
        #         return x6

class CR_FA_AdaptivePool_Net(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(CR_FA_AdaptivePool_Net, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.CR_module=ChannelRecombinationModul(2*N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.FA_module=FrequencyAttentionModul()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None,visual=False):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_cr=self.CR_module(x1)
        x_fa=self.FA_module(x_cr)
        x2=self.conv1(x_fa)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6



class TFANet_3thharmoic(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(TFANet_3thharmoic, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convm1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convm2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(3,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.FA_module=FrequencyAttentionModul()
        self.CR_module=ChannelRecombinationModul(3*N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_m=x[:,:,40:120,:]
        x_h=x[:,:,80:200,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_m=self.convm1(x_m)
        x_m=self.convm2(x_m)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_m,x_h),1)
        x_fa=self.FA_module(x1)
        x_cr=self.CR_module(x_fa)
        x2=self.conv1(x_cr)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6




class TFANet_1thharmoic(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(TFANet_1thharmoic, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.FA_module=FrequencyAttentionModul()
        self.CR_module=ChannelRecombinationModul(N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None):
        x_l=x.to(torch.float32)
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_fa=self.FA_module(x_l)
        x_cr=self.CR_module(x_fa)
        x2=self.conv1(x_cr)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6


class ChannelRecombination_AdaptivePool_Net(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(ChannelRecombination_AdaptivePool_Net, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.CR_module=ChannelRecombinationModul(2*N1,N1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_cr=self.CR_module(x1)
        x2=self.conv1(x_cr)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6





class FrequencyAttention_AdaptivePool_Net(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(FrequencyAttention_AdaptivePool_Net, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.FA_module=FrequencyAttentionModul()
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2*N1,
                out_channels=N1,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_fa=self.FA_module(x1)
        x_pw=self.pointwise_conv(x_fa)
        x2=self.conv1(x_pw)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6


class AdaptivePool_Net(nn.Module):
    def __init__(self,nb_classes,Chans,Samples,DropoutRate,N1,N2):
        super(AdaptivePool_Net, self).__init__()
        len_ratio=Samples/375
        self.convl1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
            in_channels=5*Chans,
            out_channels=N1,
            kernel_size=(5,1),
            bias=False,
            padding=(2,0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convl2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.convh1=nn.Sequential(
            nn.Conv2d(
            in_channels=Chans,
            out_channels=5*Chans,
            kernel_size=(1,25),
            bias=False,
            padding=(0,12),
            ),
            nn.Conv2d(
                in_channels=5 * Chans,
                out_channels=N1,
                kernel_size=(5, 1),
                bias=False,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2,int(15*len_ratio)),ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.convh2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3,3),
                bias=False,
                padding=(1,1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2*N1,
                out_channels=N1,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(3, 1),
                bias=False,
                padding=(1, 0),
            ),
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
            nn.Dropout(p=DropoutRate)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N1,
                kernel_size=(5, 15),
                groups=N1,
                bias=False,
                padding=(2, 7)
            ),
            nn.BatchNorm2d(N1),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=N1,
                out_channels=N2,
                kernel_size=(3, 5),
                bias=False,
                padding=(1, 2),
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.Dropout(p=DropoutRate),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=(5, 15),
                groups=N2,
                bias=False,
                padding=(2, 7),
            ),
            nn.Conv2d(
                in_channels=N2,
                out_channels=N2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(N2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 5), ceil_mode=True),
            nn.Dropout(p=DropoutRate),
        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),            #CrossEntropyLoss内置了softmax 因此此处不需额外添加
        )

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data,mode='fan_in',nonlinearity='elu')

    def forward(self,x,labels=None):
        x=x.to(torch.float32)
        x_l=x[:,:,0:40,:]
        x_h=x[:,:,40:120,:]
        x_l=self.convl1(x_l)
        x_l=self.convl2(x_l)
        x_h=self.convh1(x_h)
        x_h=self.convh2(x_h)
        x1=torch.cat((x_l,x_h),1)
        x_pw=self.pointwise_conv(x1)
        x2=self.conv1(x_pw)
        x3=self.depthwise_conv(x2)
        x4=self.conv2(x3)
        x5=self.separable_conv(x4)
        x6=self.classification(x5)
        loss_function=nn.CrossEntropyLoss()
        if labels is not None:
            loss=loss_function(x6,labels)
            return x6,loss
        else:
            return x6















