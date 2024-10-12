import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader,Dataset
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

class MyDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_list = os.listdir(self.file_path)

    def __getitem__(self, index):
        """
        Extract label from the filename.
        Please modify according to your data format.
        Assumes filename format is 'name_label.npy'.
        """
        data_title = self.data_list[index]
        label = int(data_title.split('_')[1].split('.')[0])
        data_path = os.path.join(self.file_path, data_title)
        data = np.load(data_path,allow_pickle=True)
        data = data.transpose(2, 0, 1)
        return data, label

    def __len__(self):
        return len(self.data_list)

class ChannelAttentionModul(nn.Module):
    def __init__(self, in_channel, r=0.5):
        super(ChannelAttentionModul, self).__init__()

        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_branch = self.MaxPool(x)

        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)


        avg_branch = self.AvgPool(x)

        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape

        Mc = torch.reshape(weight, (h, w, 1, 1))

        x = Mc * x

        return x

class FrequencyAttentionModul(nn.Module):
    def __init__(self):
        super(FrequencyAttentionModul, self).__init__()

    def forward(self, x):
        _, _, _, width = x.size()

        avgpool = nn.AvgPool2d((1, width))

        FrequencyWeight = avgpool(x)
        x = FrequencyWeight * x

        return x

class ChannelRecombinationModul(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelRecombinationModul, self).__init__()

        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid(),
        )

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

        max_branch = self.MaxPool(x)

        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape

        Mc = torch.reshape(weight, (h, w, 1, 1))

        x1=self.pointwise_conv(x)

        x2 = Mc * x1

        return x2


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
            nn.Linear(10*math.ceil(math.ceil(Samples/int(15*len_ratio))/5)*N2,nb_classes),
        )
    def forward(self,x):
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
        return x6

