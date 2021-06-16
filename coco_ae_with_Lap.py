import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import sys, os 
from pycocotools.coco import COCO

# from lap_pyramid_loss import LapLoss

torch.cuda.empty_cache()

# COCOデータセット
class Coco_Dataset(torch.utils.data.Dataset):  
  
    def __init__(self, data_num, root, transform=None, data_kind="train"):
        self.data_num = data_num
        # 指定する場合は前処理クラスを受け取る
        self.transform = transform[data_kind]
        self.resize64_transform = transforms.Compose([transforms.Resize(64),])
        self.resize32_transform = transforms.Compose([transforms.Resize(32),])
        self.resize16_transform = transforms.Compose([transforms.Resize(16),])
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
            # anno_path = "../coco/annotations/instances_train2014.json"
        elif data_kind == "val":
            root_path = os.path.join(root, 'val2014')
            # anno_path = "../coco/annotations/instances_val2014.json"
        else:
            root_path = os.path.join(root, 'test2014')
            # anno_path = "../coco/annotations/image_info_test2014.json"

        # coco = COCO(anno_path)

        # 画像一覧を取得
        all_images = os.listdir(root_path)
        print('root_path: ', root_path)
        print('number of all_image: ', len(all_images))
        for i in range(len(all_images)):
            self.images.append(os.path.join(root_path, all_images[i]))
            if i % 1000 == 0:
                print('load img...', i, '/', len(all_images))
            if i == self.data_num - 1:
                print(len(self.images))
                # print(self.images)
                break                  
        # for i in range(len(self.category_list)):
        #     cat_ids = coco.getCatIds(catNms=self.category_list[i])  # 指定したカテゴリに対応するcategory_IDを取得する
        #     img_ids = coco.getImgIds(catIds=cat_ids)  # 指定したカテゴリ ID の物体がすべて存在する画像の ID 一覧を取得する。

        #     # 指定した画像 ID に対応する画像情報とラベルを取得する。
        #     for j in range(len(img_ids)):
        #         img_info = coco.loadImgs(img_ids)
        #         if j % 1000 == 0:
        #             print('load img', j, '/', len(img_ids))
        #         self.images.append(os.path.join(root_path, '/', img_info[j]["file_name"]))
        #         self.labels.append(self.category_list[i])
        
    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得
        image = self.images[index]
        # label = self.labels[index]
        # 画像ファイルパスから画像を読み込む
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        # 前処理がある場合は前処理をいれる
        if self.transform is not None:
            image = self.transform(image)
            resize64_image = self.resize64_transform(image)
            resize32_image = self.resize32_transform(image)
            resize16_image = self.resize16_transform(image)
        # 画像とラベルのペアを返却
        # return image, label
        return image, resize64_image, resize32_image, resize16_image
        
    def __len__(self):
        # ここにはデータ数を指定
        return len(self.images)


class CNN_AutoEncoder(nn.Module):

    def __init__(self):
        super(CNN_AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(  # in(3*256*256)
            nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=5),  # out(16*64*64)
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # out(16*32*32)
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),  # out(16*32*32)
            nn.Conv2d(16, 8, kernel_size=5, stride=2, padding=2),  # out(8*16*16)
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # out(8*8*8)
            nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2),  # out(8*8*8)
        )
        self.Decoder1 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),  # out(8*16*16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 16, kernel_size=4, stride=4),  # out(16*64*64)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=4),  # out(3*256*256)
            # nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        self.Decoder2 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),  # out(8*16*16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=4),  # out(3*64*64)
            # nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        self.Decoder3 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),  # out(8*16*16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2),  # out(3*32*32)
            # nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        self.Decoder4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2),  # out(3*16*16)
            # nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 512),
        )

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=5)  # out(16*64*64)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # out(16*32*32)
        # self.conv2 = nn.Conv2d(16, 8, kernel_size=5, stride=2, padding=2)  # out(8*16*16)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # out(8*8*8)

        # self.t_conv1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)  # out(8*16*16)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.t_conv2 = nn.ConvTranspose2d(8, 16, kernel_size=4, stride=4)  # out(16*64*64)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.t_conv3 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=4)  # out(3*256*256)
        # self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        enc_x = self.Encoder(x)

        # mid_x = enc_x.reshape(-1, 512)
        # mid_x = self.fc(mid_x)
        # mid_x = mid_x.reshape(-1, 8, 8, 8)
        mid_x = enc_x

        dec1_x = self.Decoder1(mid_x)
        dec2_x = self.Decoder2(mid_x)
        dec3_x = self.Decoder3(mid_x)
        dec4_x = self.Decoder4(mid_x)

        # dec1_edge = make_edge(dec1_x)
        # dec2_edge = make_edge(dec2_x)
        # dec3_edge = make_edge(dec3_x)

        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool2(x)

        # x = self.t_conv1(x)
        # x = self.relu3(x)
        # x = self.t_conv2(x)
        # x = self.relu4(x)
        # x = self.t_conv3(x)
        # x = self.relu5(x)

        # return dec1_edge, dec2_edge, dec3_edge
        return dec1_x, dec2_x, dec3_x, dec4_x

def normalize_edge(diff):
    result = []
    # for diff in images:
    diff_min = torch.min(diff)
    diff_max = torch.max(diff)
    diff_normalize = (diff - diff_min) / (diff_max - diff_min)
        # result.append(diff_normalize)
    # return result
    return diff_normalize

def make_edge(images, rate):
    # downsample_func = nn.MaxPool2d(kernel_size=2, stride=2)  # 画像サイズをダウンサンプリング
    downsample_func = nn.AvgPool2d(kernel_size=rate, stride=rate)  # 画像サイズをダウンサンプリング
    upsample_func = nn.Upsample(scale_factor=rate, mode='bilinear', align_corners=True)  # 元のサイズに戻してぼやけ画像を取得

    downsample_images = downsample_func(images)
    upsample_images = upsample_func(downsample_images)
    edge = images - upsample_images  # 元画像とぼやけ画像の差分をとるとエッジを抽出できる
    # edge = upsample_images - images  # 元画像とぼやけ画像の差分をとるとエッジを抽出できる
    # edge = torch.trunc(edge * 255)
    # edge = normalize_edge(edge)
    return edge

def laploss(output_image, input_image, criterion, stages):
    input_edge = []
    output_edge = []
    loss = 0
    for i in range(1, stages+1):
        input_edge.append(make_edge(input_image, 2**i))
        output_edge.append(make_edge(output_image, 2**i))
        loss += criterion(input_edge[i-1], output_edge[i-1])

    # output_edge = output_edge.to('cpu')  # ---
    # show_image(output_edge.reshape(-1, 3, 256, 256), image_flag="--")  # ---
    # loss_2 = criterion(output_edge_2, input_edge_2)
    # loss_4 = criterion(output_edge_4, input_edge_4)
    # loss_8 = criterion(output_edge_8, input_edge_8)
    # loss_16 = criterion(output_edge_16, input_edge_16)
    # loss_32 = criterion(output_edge_32, input_edge_32)
    # loss_64 = criterion(output_edge_64, input_edge_64)
    # loss = loss_2 + loss_4 + loss_8 + loss_16 + loss_32 + loss_64
    return loss

def try_show_image(image):
    image = image.to('cpu')
    show_image(image.reshape(-1, 3, 256, 256), image_flag="--")
    # sys.exit()

def training(train_loader, model, criterion, optimizer, device):
    train_loss = 0
    # train_acc = 0

    for i, (images, resize64_images, resize32_images, resize16_images) in enumerate(train_loader): 
        images = images.to(device)
        resize64_images = resize64_images.to(device)
        resize32_images = resize32_images.to(device)
        resize16_images = resize16_images.to(device)
        
        model.zero_grad()
        outputs, r64_outputs, r32_outputs, r16_outputs = model(images)
        
        # loss = criterion(outputs, images)
        # loss_r64 = criterion(r64_outputs, resize64_images)
        # loss_r16 = criterion(r16_outputs, resize16_images)
        loss = laploss(outputs, images, criterion, stages=6)
        loss_r64 = laploss(r64_outputs, resize64_images, criterion, stages=5)
        loss_r32 = laploss(r32_outputs, resize32_images, criterion, stages=4)
        loss_r16 = laploss(r16_outputs, resize16_images, criterion, stages=3)
        loss = loss + loss_r64 + loss_r32 + loss_r16

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # train_acc += (outputs.max(1)[1] == labels).sum().item()  #
    
    # try_show_image(make_edge(outputs, 64))
    # try_show_image(outputs)

    ave_train_loss = train_loss / len(train_loader.dataset)
    # ave_train_acc = train_acc / len(train_loader.dataset)
    return ave_train_loss

def testing(test_loader, model, criterion, optimizer, device):
    val_loss = 0
    # val_acc = 0
    outputs_and_inputs = []
    
    for images, resize64_images, resize32_images, resize16_images in test_loader:
        images = images.to(device)
        resize64_images = resize64_images.to(device)
        resize32_images = resize32_images.to(device)
        resize16_images = resize16_images.to(device)

        outputs, r64_outputs, r32_outputs, r16_outputs = model(images)

        # loss = criterion(outputs, images)
        # loss_r64 = criterion(r64_outputs, resize64_images)
        # loss_r16 = criterion(r16_outputs, resize16_images)
        loss = laploss(outputs, images, criterion, stages=6)
        loss_r64 = laploss(r64_outputs, resize64_images, criterion, stages=5)
        loss_r32 = laploss(r32_outputs, resize32_images, criterion, stages=4)
        loss_r16 = laploss(r16_outputs, resize16_images, criterion, stages=3)
        loss = loss + loss_r64 + loss_r32 + loss_r16

        val_loss += loss.item()
        # val_acc += (outputs.max(1)[1] == labels).sum().item()  #
        outputs_and_inputs.append((outputs, images))
       
    ave_val_loss = val_loss / len(test_loader.dataset)
    # ave_val_acc = val_acc / len(test_loader.dataset)
    return ave_val_loss, outputs_and_inputs

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
    loss_fig.savefig(path + "coco_AutoEncoder_" + draw_flag + "_lap_0615.png")
    plt.show()

def show_image(img, image_flag):
    path = 'movies/'
    img = torchvision.utils.make_grid(img)
    # torchvision.utils.save_image(img, "coco_AutoEncoder_" + image_flag + "_0607.png")
    img = img / 2 + 0.5
    if image_flag == "out":
        img = img.mul(torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1))
        img = img.add(torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1))
    npimg = img.detach().numpy()
    figure_image = plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    figure_image.savefig(path + "coco_AutoEncoder_" + image_flag + "_lap_0615.png")
    plt.show()

def main():
    num_epoch = 30
    num_batch = 64
    data_train_num = 2000
    data_val_num = 500
    data_test_num = 500
    train_loss_list = []
    # train_acc_list = []
    val_loss_list = []
    # val_acc_list = []
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
    train_dataset = Coco_Dataset(data_train_num, root, data_transforms, data_kind='train')
    val_dataset = Coco_Dataset(data_val_num, root, data_transforms, data_kind='val')
    test_dataset = Coco_Dataset(data_test_num, root, data_transforms, data_kind='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN_AutoEncoder().to(device)
    print(device)  # GPUを使えているか
    print(model)  # ネットワーク構造を記述


    criterion = torch.nn.MSELoss()
    # criterion = LapLoss(max_levels=3, channels=3, device=device)
    # criterion = LapLoss(max_levels=7, channels=3, device=device)
    # criterion2 = LapLoss(max_levels=1, channels=3, device=device)
    # criterion3 = LapLoss(max_levels=1, channels=3, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   #adam  lr=0.0001

    print('Start training...')

    for epoch in range(num_epoch):
        # train
        model.train()
        ave_train_loss = training(train_loader, model, criterion, optimizer, device)

        # eval
        model.eval()
        ave_val_loss, _ = testing(val_loader, model, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {ave_train_loss:.5f},",
            f"val_loss: {ave_val_loss:.5f}")

        # record losses
        train_loss_list.append(ave_train_loss)
        # train_acc_list.append(ave_train_acc)
        val_loss_list.append(ave_val_loss)
        # val_acc_list.append(ave_val_acc)

        # save parameters of the model
        if is_save == True:
            if (epoch+1) % 100 == 0:
                model_path = 'model_ae_lap_' + str(epoch+1) + '.pth'
                optim_path = 'optim_ae_lap_' + str(epoch+1) + '.pth'
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
    
    drawing_graph(num_epoch, train_loss_list, val_loss_list, draw_flag="loss")
    # drawing_graph(num_epoch, train_acc_list, val_acc_list, draw_flag="accuracy")

    # save parameters of the model
    if is_save == True:
        model_path = 'model_ae_lap_' + str(epoch+1) + '.pth'
        optim_path = 'optim_ae_lap_' + str(epoch+1) + '.pth'
        # model_path = 'model_ae.pth'
        # optim_path = 'optim_ae.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    model2 = CNN_AutoEncoder().to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)   #adam  lr=0.0001
    # read parameters of the model
    # model_path = 'model_ae_50.pth'
    model_path = 'model_ae_lap_' + str(epoch+1) + '.pth'
    model2.load_state_dict(torch.load(model_path))
    # optimizer2.load_state_dict(torch.load(optim_path))

    # test
    model2.eval()
    print('Test begin...')
    ave_test_loss, outputs_and_inputs = testing(test_loader, model, criterion, optimizer, device)
    print(f"Test Loss: {ave_test_loss:.5f}")
    # 入力画像と出力画像を表示
    output_image, input_image = outputs_and_inputs[-1]
    output_image = output_image.to('cpu')
    input_image = input_image.to('cpu')
    show_image(input_image.reshape(-1, 3, 256, 256), image_flag="in")
    show_image(output_image.reshape(-1, 3, 256, 256), image_flag="out")

if __name__ == "__main__":
    main()