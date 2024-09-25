from sklearn.model_selection import KFold
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from MyNet import TFA_Net
from MyNet import MyDataset
from MyNet import setup_seed


seed=42
setup_seed(seed)
device=torch.device('cuda')
BatchSize=64
epochs=300
K=10

mydataset=MyDataset(r'/home/xl2/MyDataset/ssvep_data_wt_3thHarmonic_8chans_1s/')
kfold = KFold(n_splits=K, shuffle=True,random_state=seed)
kfold_gen=kfold.split(mydataset)

val_size=int(len(mydataset)/K)
train_size=len(mydataset)-val_size


#################搜寻断点路径的函数####################
def SearchCheckpoint(checkpoint_dir,current_num):
    for epoch in range(300,0,-30):
        checkpoint_name = f"{current_num}_{epoch}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
    return None
###################################################

#########跳过已完成的交叉检验次数########
already_done=6  ##############此参数修改
current_num=1
for i in range(0,already_done):
    next(kfold_gen)
    current_num+=1
####################################

for train_indices, val_indices in kfold_gen:

    train_set = torch.utils.data.Subset(mydataset, train_indices)
    val_set = torch.utils.data.Subset(mydataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BatchSize, shuffle=True,num_workers=16,drop_last=True,pin_memory=True,prefetch_factor=4,persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BatchSize, shuffle=False,num_workers=16,drop_last=True)

    mymodel = TFANet_3thharmoic(40, 8, 250, 0.7, 100, 90)
    mymodel = mymodel.to(device)
    mymodel = nn.DataParallel(mymodel)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in mymodel.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001, weight_decay=0.00001)

    start_epoch=0
    loss_list = []
    accuracy_list = []

    ########################################搜寻是否有断点路径#############################################################
    checkpoint_dir='/home/xl2/model_checkpoint/10Fold_cross_val/TFANet_3thharmoic_8chans_1s/'
    path_checkpoint=SearchCheckpoint(checkpoint_dir=checkpoint_dir,current_num=current_num)
    if path_checkpoint is not None:
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        mymodel.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        loss_list = checkpoint['loss']
        accuracy_list = checkpoint['acc']
    ###################################################################################################################

    for i in mymodel.children():
        print(i)
    summary(mymodel, (8, 200, 250), BatchSize, device="cuda")

    print(current_num)

    mymodel.train()
    for epoch in range(start_epoch + 1, epochs + 1):
        loss_epoch = 0
        accuracy_epoch = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data, target = data.to(device), target.to(device)
                # data=torch.permute(data,(0,3,1,2))

                optimizer.zero_grad()
                outputs, loss = mymodel(data, target)
                loss = loss.mean()
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == target).sum().item()
                accuracy = correct / BatchSize

                loss_epoch += loss.cpu().detach().numpy()
                accuracy_epoch += correct

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        loss_list.append(loss_epoch / (train_size / BatchSize))
        accuracy_list.append(accuracy_epoch / train_size)
        if (epoch == epochs):
            checkpoint = {
                "net": mymodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss_list,
                "acc": accuracy_list
            }
            torch.save(checkpoint, f"/home/xl2/checkpoint/10Fold_cross_val/TFANet_3thharmoic_8chans_1s/{current_num}.pth")

        elif (epoch % 30) == 0:
            checkpoint = {
                "net": mymodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss_list,
                "acc": accuracy_list
            }
            torch.save(checkpoint, f"/home/xl2/model_checkpoint/10Fold_cross_val/TFANet_3thharmoic_8chans_1s/{current_num}_{epoch}.pth")

    print(accuracy_list[-1])
    plot_x = range(1, epochs + 1)
    plot_y1 = loss_list
    plot_y2 = accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(plot_x, plot_y1)
    plt.subplot(2, 1, 2)
    plt.plot(plot_x, plot_y2)
    plt.show()

    mymodel.eval()
    with torch.no_grad():
        total_acc = 0
        for val_images, val_labels in val_loader:
            outputs = mymodel(val_images.to(device))
            predictions_y = torch.max(outputs, dim=1)[1]
            correct = (predictions_y == val_labels.to(device)).sum().item()
            total_acc += correct
        print(total_acc / val_size)

    current_num+=1