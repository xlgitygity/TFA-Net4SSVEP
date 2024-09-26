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
from Proposed_Model import TFANet
from Control_Model import DeepConvNet, EEGNet, CCNN
from Proposed_Model import MyDataset
from Proposed_Model import setup_seed

seed=42
setup_seed(seed)
device=torch.device('cuda')
BatchSize=64
epochs=300
K=10

dataset = MyDataset(r'path/to/your/dataset/')
kfold = KFold(n_splits=K, shuffle=True,random_state=seed)
kfold_gen=kfold.split(dataset)

val_size=int(len(dataset)/K)
train_size=len(dataset)-val_size

################# Function to search for checkpoint paths ####################
def SearchCheckpoint(checkpoint_dir,current_num):
    for epoch in range(epochs,0,-30):
        checkpoint_name = f"{current_num}_{epoch}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
    return None
###################################################

######### Skip completed cross-validation iterations ########
already_done=0
current_num=1
for i in range(0,already_done):
    next(kfold_gen)
    current_num+=1
####################################

for train_indices, val_indices in kfold_gen:

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BatchSize, shuffle=True,num_workers=16,drop_last=True,pin_memory=True,prefetch_factor=4,persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BatchSize, shuffle=False,num_workers=16,drop_last=True)

    ######### Control model configurations #########
    # model = DeepConvNet(40, 8, 250)
    # model = EEGNet(40, 8, 250, 0.3)
    # CNN_PARAMS = {              # C-CNN parameter settings.
    #     'batch_size': 64,       # For details, please refer to https://github.com/aaravindravi/Brain-computer-interfaces
    #     'epochs': 50,
    #     'droprate': 0.25,
    #     'learning_rate': 0.001,
    #     'weight_decay':0.00001,
    #     'kernel_f': 10,
    #     'n_ch': 8,
    #     'n_fc':300,
    #     'num_classes': 40}
    # model = CCNN(CNN_PARAMS)
    ###############################################

    ######## Proposed model configurations #######
    model = TFANet(40, 8, 250, 0.7, 100, 90)   # Example with one second of data
    model = model.to(device)
    model = nn.DataParallel(model)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001, weight_decay=0.00001)

    start_epoch=0
    loss_list = []
    accuracy_list = []

    ######################################## Check if a checkpoint path exists #############################################################
    checkpoint_dir = 'path/to/your/checkpoint/directory/'
    path_checkpoint = SearchCheckpoint(checkpoint_dir=checkpoint_dir, current_num=current_num)
    if path_checkpoint is not None:
        checkpoint = torch.load(path_checkpoint)  # Load the checkpoint
        model.load_state_dict(checkpoint['net'])  # Load model learnable parameters
        optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer parameters
        start_epoch = checkpoint['epoch']  # Set the starting epoch
        loss_list = checkpoint['loss']
        accuracy_list = checkpoint['acc']
    ###################################################################################################################

    for i in model.children():
        print(i)
    summary(model, (8, 120, 250), BatchSize, device="cuda")  #[channels,frequency,time]

    print(current_num)

    criterion=nn.CrossEntropyLoss.cuda()
    model.train()
    for epoch in range(start_epoch + 1, epochs + 1):
        loss_epoch = 0
        accuracy_epoch = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for data, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data, labels = data.to(device), labels.to(device)

                outputs = model(data, labels)

                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == labels).sum().item()
                accuracy = correct / BatchSize

                loss_epoch += loss.cpu().detach().numpy()
                accuracy_epoch += correct

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        loss_list.append(loss_epoch / (train_size / BatchSize))
        accuracy_list.append(accuracy_epoch / train_size)

        # Save the model in the checkpoint folder after completing the iteration
        if (epoch == epochs):
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss_list,
                "acc": accuracy_list
            }
            torch.save(checkpoint, f"path/to/your/checkpoint/directory/{current_num}.pth")

        # Save the model in the model_checkpoint folder every 30 iterations
        elif (epoch % 30) == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss_list,
                "acc": accuracy_list
            }
            torch.save(checkpoint, f"path/to/your/model_checkpoint/directory/{current_num}_{epoch}.pth")

    print(accuracy_list[-1])
    plot_x = range(1, epochs + 1)
    plot_y1 = loss_list
    plot_y2 = accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(plot_x, plot_y1)
    plt.subplot(2, 1, 2)
    plt.plot(plot_x, plot_y2)
    plt.show()

    model.eval()
    with torch.no_grad():
        total_acc = 0
        for val_data, val_labels in val_loader:
            outputs = model(val_data.to(device))
            predictions_y = torch.max(outputs, dim=1)[1]
            correct = (predictions_y == val_labels.to(device)).sum().item()
            total_acc += correct
        print(total_acc / val_size)

    # Complete one cross-validation iteration, increment current_num by 1
    current_num+=1
