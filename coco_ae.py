import numpy as np
import matplotlib.pyplot as plt
import torch
import sys, os 
from pycocotools.coco import COCO

# anno_path = "coco/annotations/instances_train2014.json"
# coco = COCO(anno_path)

# # try
# cat_ids = coco.getCatIds(catNms=["dog", "cat"])  # 指定したカテゴリに対応するcategory_IDを取得する
# # print(coco.getCatIds(supNms=["vehicle"]))
# img_ids = coco.getImgIds(catIds=cat_ids)  # 指定したカテゴリ ID の物体がすべて存在する画像の ID 一覧を取得する。

# COCOデータセット
class Coco_Dataset(torch.utils.data.Dataset):  
    classes = ['ant', 'bee']
  
    def __init__(self, root, transform=None, data_kind="train"):
        # 指定する場合は前処理クラスを受け取る
        self.transform = transform[data_kind]
        # label: 80種類
        self.category_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        # 画像を読み込むファイルパスとラベルのリスト
        self.images = []
        self.labels = []
        # ルートフォルダーパス
        # root = "hymenoptera_data"
        # 訓練の場合と検証の場合でフォルダわけ
        # 画像を読み込むファイルパスを取得
        if data_kind == "train":
            root_path = os.path.join(root, 'train2014')
            anno_path = "coco/annotations/instances_train2014.json"
        elif data_kind == "val":
            root_path = os.path.join(root, 'val2014')
            anno_path = "coco/annotations/instances_val2014.json"
        else:
            root_path = os.path.join(root, 'test2014')
            anno_path = "coco/annotations/instances_test2014.json"

        coco = COCO(anno_path)

        # 画像一覧を取得
        all_images = os.listdir(root_path)
        for i in range(len(self.category_list)):
            cat_ids = coco.getCatIds(catNms=self.category_list[i])  # 指定したカテゴリに対応するcategory_IDを取得する
            img_ids = coco.getImgIds(catIds=cat_ids)  # 指定したカテゴリ ID の物体がすべて存在する画像の ID 一覧を取得する。

            # 指定した画像 ID に対応する画像情報とラベルを取得する。
            for j in range(len(img_ids)):
                img_info, = coco.loadImgs(img_id)
                self.images.append(os.path.join(root_path, '/', img_info["file_name"]))
                self.labels.append(self.category_list[i])
        
    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得
        image = self.images[index]
        label = self.labels[index]
        # 画像ファイルパスから画像を読み込む
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        # 前処理がある場合は前処理をいれる
        if self.transform is not None:
            image = self.transform(image)
        # 画像とラベルのペアを返却
        # return image, label
        return image
        
    def __len__(self):
        # ここにはデータ数を指定
        return len(self.images)


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(  # in(3*256*256)
            nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=5),  # out(16*64*64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # out(16*32*32)
            nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2),  # out(8*16*16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # out(8*8*8)
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),  # out(8*16*16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 16, kernel_size=4, stride=4),  # out(16*64*64)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=4),  # out(3*256*256)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.Encoder(x)
        # x = x.view(x.size(0), -1)  # flatten
        x = self.Decoder(x)
        return x


def training(train_loader, model, criterion, optimizer, device):
    train_loss = 0
    train_acc = 0

    for i, images in enumerate(train_loader): 
        images = images.to(device)
        
        model.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # train_acc += (outputs.max(1)[1] == labels).sum().item()  #

    ave_train_loss = train_loss / len(train_loader.dataset)
    # ave_train_acc = train_acc / len(train_loader.dataset)
    return ave_train_loss

def testing(test_loader, model, criterion, optimizer, device):
    val_loss = 0
    val_acc = 0
    
    for images in test_loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)
        val_loss += loss.item()
        # val_acc += (outputs.max(1)[1] == labels).sum().item()  #
       
    ave_val_loss = val_loss / len(test_loader.dataset)
    # ave_val_acc = val_acc / len(test_loader.dataset)
    return ave_val_loss

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
    loss_fig.savefig(path + "coco_AutoEncoder_" + draw_flag + "_0607.png")
    plt.show()

def main():
    num_epoch = 100
    num_batch = 128
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    is_save = True  # save the model parameters 

    #画像の前処理を定義
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),  # ランダムにトリミングして (256, 256)の形状にしてる
            transforms.RandomHorizontalFlip(),  # 50%の確率で水平方向に反転させる
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # ランダムに明るさ、コントラスト、彩度、色相を変化させる
            transforms.ToTensor(),  # Tensorに変換
            # transforms.RandomErasing(),  # ランダムにカットアウトさせる
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
        'val': transforms.Compose([
            transforms.Resize(289),  # 画像のサイズを(289, 289)にする
            transforms.CenterCrop(256),  # (256, 256)にするために、サイズ変更された画像を中央で切り取る
            transforms.ToTensor(),  # Tensorに変換
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
        'test': transforms.Compose([
            transforms.Resize(289),  # 画像のサイズを(289, 289)にする
            transforms.CenterCrop(256),  # (256, 256)にするために、サイズ変更された画像を中央で切り取る
            transforms.ToTensor(),  # Tensorに変換
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
    }
    
    root = '../coco/images'
    train_dataset = Coco_Dataset(root, data_transforms, data_kind='train')
    val_dataset = Coco_Dataset(root, data_transforms, data_kind='val')
    test_dataset = Coco_Dataset(root, data_transforms, data_kind='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoder().to(device)
    print(device)  # GPUを使えているか
    print(model)  # ネットワーク構造を記述


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   #adam  lr=0.0001

    for epoch in range(num_epoch):
        # train
        model.train()
        ave_train_loss = training(train_loader, model, criterion, optimizer, device)

        # eval
        model.eval()
        ave_val_loss = testing(val_loader, model, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {ave_train_loss:.5f},"
            f"val_loss: {ave_val_loss:.5f}")

        # record losses
        train_loss_list.append(ave_train_loss)
        # train_acc_list.append(ave_train_acc)
        val_loss_list.append(ave_val_loss)
        # val_acc_list.append(ave_val_acc)
    
    drawing_graph(num_epoch, train_loss_list, val_loss_list, draw_flag="loss")
    # drawing_graph(num_epoch, train_acc_list, val_acc_list, draw_flag="accuracy")

    # save parameters of the model
    if is_save == True:
        model_path = 'model_cnn.pth'
        optim_path = 'optim_cnn.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    model2 = Net(num_classes)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)   #adam  lr=0.0001
    # read parameters of the model
    model_path = 'model_cnn.pth'
    model2.load_state_dict(torch.load(model_path))
    # optimizer2.load_state_dict(torch.load(optim_path))

    # test
    model2.eval()
    ave_test_loss = testing(test_loader, model, criterion, optimizer, device)
    print(f"Test Loss: {ave_test_loss:.5f}, Test Acc: {ave_test_acc:.5f}")

if __name__ == "__main__":
    main()