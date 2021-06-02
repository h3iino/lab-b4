import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys, os 

# CNN class
class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),  # Conv2d(入力チャンネル数, 出力チャンネル数, カーネルサイズ, ストライド, パディング)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  #
            nn.ReLU(inplace=True),  #
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


def training(train_loader, model, criterion, optimizer, device):
    train_loss = 0
    train_acc = 0

    for i, (images, labels) in enumerate(train_loader): 
        images, labels = images.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()  #

    ave_train_loss = train_loss / len(train_loader.dataset)
    ave_train_acc = train_acc / len(train_loader.dataset)
    return ave_train_loss, ave_train_acc

def testing(test_loader, model, criterion, optimizer, device):
    val_loss = 0
    val_acc = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += (outputs.max(1)[1] == labels).sum().item()  #
       
    ave_val_loss = val_loss / len(test_loader.dataset)
    ave_val_acc = val_acc / len(test_loader.dataset)
    return ave_val_loss, ave_val_acc

def drawing_graph(num_epoch, train_loss_list, val_loss_list, draw_flag="loss"):
    path = 'movies/'
    loss_fig = plt.figure()
    plt.plot(range(num_epoch), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epoch), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation ' + draw_flag)
    plt.grid()
    loss_fig.savefig(path + "cifar10_cnn_" + draw_flag + "_0601.png")
    plt.show()

def main():
    num_epoch = 150
    num_batch = 256
    num_classes = 10
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    is_save = True  # save the model parameters 

    #画像の前処理を定義
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(29),  # ランダムにトリミングして (29, 29)の形状に
            transforms.RandomHorizontalFlip(),  # 50%の確率で水平方向に反転させる
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # ランダムに明るさ、コントラスト、彩度、色相を変化
            transforms.ToTensor(),  # Tensorに変換
            transforms.RandomErasing(),  # ランダムにカットアウト
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
        'val': transforms.Compose([
            transforms.Resize(29),  # 画像のサイズを(29, 29)にする
            # transforms.CenterCrop(29),  # (29, 29)にするために、サイズ変更された画像を中央で切り取る
            transforms.ToTensor(),  # Tensorに変換
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
    }

    train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=True, 
                                                transform=data_transforms['train'],
                                                download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=False, 
                                                transform=data_transforms['val'],
                                                download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(num_classes).to(device)
    print(device)  # GPUを使えているか
    print(model)  # ネットワーク構造を記述


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   #adam  lr=0.0001

    for epoch in range(num_epoch):
        # train
        model.train()
        ave_train_loss, ave_train_acc = training(train_loader, model, criterion, optimizer, device)

        # eval
        model.eval()
        ave_val_loss, ave_val_acc = testing(test_loader, model, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {ave_train_loss:.5f},"
            f"acc: {ave_train_acc:.5f}, val_loss: {ave_val_loss:.5f}, "
            f"val_acc: {ave_val_acc:.5f}")

        # record losses
        train_loss_list.append(ave_train_loss)
        train_acc_list.append(ave_train_acc)
        val_loss_list.append(ave_val_loss)
        val_acc_list.append(ave_val_acc)
    
    drawing_graph(num_epoch, train_loss_list, val_loss_list, draw_flag="loss")
    drawing_graph(num_epoch, train_acc_list, val_acc_list, draw_flag="accuracy")

    # save parameters of the model
    if is_save == True:
        model_path = 'model_cnn.pth'
        optim_path = 'optim_cnn.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    model2 = Net(num_classes)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)   #adam  lr=0.0001
    # read parameters of the model
    model_path = 'model_cnn.pth'
    model2.load_state_dict(torch.load(model_path))

    # test
    model2.eval()
    ave_test_loss, ave_test_acc = testing(test_loader, model, criterion, optimizer, device)
    print(f"Test Loss: {ave_test_loss:.5f}, Test Acc: {ave_test_acc:.5f}")

if __name__ == "__main__":
    main()