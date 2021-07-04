import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import sys, os 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from pycocotools.coco import COCO

torch.cuda.empty_cache()

# anno_path = "coco/annotations/instances_train2014.json"
# coco = COCO(anno_path)

# # try
# cat_ids = coco.getCatIds(catNms=["dog", "cat"])  # 指定したカテゴリに対応するcategory_IDを取得する
# # print(coco.getCatIds(supNms=["vehicle"]))
# img_ids = coco.getImgIds(catIds=cat_ids)  # 指定したカテゴリ ID の物体がすべて存在する画像の ID 一覧を取得する。


# COCOデータセット
class Coco_Dataset(torch.utils.data.Dataset):  
  
    def __init__(self, data_num, root, transform=None, data_kind="train"):
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
            if i == data_num-1:
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
        # 画像とラベルのペアを返却
        # return image, label
        return image
        
    def __len__(self):
        # ここにはデータ数を指定
        return len(self.images)



class CNN_AutoEncoder(nn.Module):

    def __init__(self):
        super(CNN_AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(  # in(3*256*256)
            # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),  # out(8*64*64)
            nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2),  # out(128*128*128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2),  # out(64*64*64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),  # out(32*32*32)
            # nn.Conv2d(16, 64, kernel_size=5, stride=4, padding=2),  # out(16*16*16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),  # out(4*16*16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),  # out(16*8*8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),  # out(16*4*4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # out(16*8*8)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # out(16*16*16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),  # out(64*32*32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # out(64*64*64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),  # out(128*128*128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2),  # out(3*256*256)
            nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            # nn.Linear(16, 8),
            # nn.BatchNorm2d(8),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            # nn.Linear(8, 8),
            # nn.BatchNorm2d(8),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            # nn.Linear(8, 16),
            # nn.BatchNorm2d(8),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        # self.fc1 = nn.Linear(8, 4)
        # self.bn1 = nn.BatchNorm2d(4)
        self.rl1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        # # self.fc2 = nn.Linear(4, 4)
        # # self.bn2 = nn.BatchNorm2d(4)
        self.rl2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.fc3 = nn.Linear(4, 8)
        # self.bn3 = nn.BatchNorm2d(4)
        self.rl3 = nn.ReLU(inplace=True)


    def forward(self, x):
        batch_size = x.shape[0]
        enc_x = self.Encoder(x)
        # print(enc_x.shape)

        # x = x.reshape(-1, 512)
        # mid_x = enc_x.view(-1, 512)
        mid_x = enc_x.clone()
        mid_x = mid_x.view(batch_size, -1)
        # print("e", enc_x.dtype)
        # x = self.fc(x)
        # mid_x = self.rl1(self.bn1(self.fc1(mid_x)))
        mid_x = self.rl2(self.bn2(self.fc2(mid_x)))
        # mid_x = self.rl3(self.bn3(self.fc3(mid_x)))
        # print("m", mid_x.shape)
        dec_x = mid_x.clone()
        dec_x = dec_x.view(batch_size, 32, 4, 4)
        # dec_x = dec_x.view(batch_size, 2, 16, 16)
        # print("d", dec_x.shape)

        dec_x = self.Decoder(dec_x)

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

        return dec_x


def training(train_loader, model, criterion, optimizer, device, model_flag):
    train_loss = 0
    # train_acc = 0

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

def testing(test_loader, model, criterion, optimizer, device, model_flag):
    val_loss = 0
    # val_acc = 0
    outputs_and_inputs = []
    
    for images in test_loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)
        # loss = np.average(np.array(images)**2 - np.array(outputs)**2)
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
    loss_fig.savefig(path + "coco_AutoEncoder_" + draw_flag + "_0702.png")
    # loss_fig.savefig(path + "coco_AutoEncoder_" + draw_flag + "_0615.png")
    plt.show()

# Min-Maxスケーリング
def normalize_images(images):
    result = []
    for diff in images:
        diff_min = torch.min(diff)
        diff_max = torch.max(diff)
        diff_normalize = (diff - diff_min) / (diff_max - diff_min)
        result.append(diff_normalize)
    return result

def show_image(img, image_flag):
    path = 'movies/'
    img = torchvision.utils.make_grid(img)
    # torchvision.utils.save_image(img, "coco_AutoEncoder_" + image_flag + "_0607.png")
    img = img / 2 + 0.5
    # npimg = np.clip(npimg, 0, 1)
    # if image_flag == "out":
    #     # img = normalize_images(img)
    #     img = img.mul(torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1))
    #     img = img.add(torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1))
    npimg = img.detach().numpy()
    figure_image = plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    cv2.imshow(npimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imsave(path + "coco_AutoEncoder_" + image_flag + "_sample_0702.png", npimg)
    # figure_image.subplots_adjust(left=0, right=0, bottom=0, top=0)

    figure_image.savefig(path + "coco_AutoEncoder_" + image_flag + "_sample_0702.png")
    # figure_image.savefig(path + "coco_AutoEncoder_" + image_flag + "_0615.png")
    plt.show()
    return npimg

def main():
    num_epoch = 300
    # num_batch = 128
    num_batch = 4
    data_train_num = 2000
    data_val_num = 50
    data_test_num = 50
    train_loss_list = []
    # train_acc_list = []
    val_loss_list = []
    # val_acc_list = []
    is_save = True  # save the model parameters 
    model_flag = "cnn"
    # model_flag = "linear"

    #画像の前処理を定義
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(256),  # ランダムにトリミングして (256, 256)の形状にしてる
            transforms.Resize(256),  # 画像のサイズを(256, 256)にする
            transforms.CenterCrop(256),  # (256, 256)にするために、サイズ変更された画像を中央で切り取る
            # transforms.RandomHorizontalFlip(),  # 50%の確率で水平方向に反転させる
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # ランダムに明るさ、コントラスト、彩度、色相を変化させる
            transforms.ToTensor(),  # Tensorに変換
            # transforms.RandomErasing(),  # ランダムにカットアウトさせる
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # 画像のサイズを(289, 289)にする
            transforms.CenterCrop(256),  # (256, 256)にするために、サイズ変更された画像を中央で切り取る
            transforms.ToTensor(),  # Tensorに変換
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),  # 画像のサイズを(289, 289)にする
            transforms.CenterCrop(256),  # (256, 256)にするために、サイズ変更された画像を中央で切り取る
            transforms.ToTensor(),  # Tensorに変換
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
        ]),
    }
    
    root = '../coco/images'
    train_dataset = Coco_Dataset(data_train_num, root, data_transforms, data_kind='train')
    val_dataset = Coco_Dataset(data_val_num, root, data_transforms, data_kind='val')
    # test_dataset = Coco_Dataset(data_test_num, root, data_transforms, data_kind='test')
    test_dataset = Coco_Dataset(data_test_num, root, data_transforms, data_kind='train')  # 画像を再構成できているか確認するため

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)  # 画像を再構成できているか確認するため
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if model_flag == "cnn":
    model = CNN_AutoEncoder().to(device)

    print(device)  # GPUを使えているか
    print(model)  # ネットワーク構造を記述


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   #adam  lr=0.0001

    # print('Start training...')

    # for epoch in range(num_epoch):
    #     # train
    #     model.train()
    #     ave_train_loss = training(train_loader, model, criterion, optimizer, device, model_flag)

    #     # eval
    #     model.eval()
    #     ave_val_loss, _ = testing(val_loader, model, criterion, optimizer, device, model_flag)
    #     print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {ave_train_loss:.5f},",
    #         f"val_loss: {ave_val_loss:.5f}")

    #     # record losses
    #     train_loss_list.append(ave_train_loss)
    #     # train_acc_list.append(ave_train_acc)
    #     val_loss_list.append(ave_val_loss)
    #     # val_acc_list.append(ave_val_acc)

    #     # save parameters of the model
    #     if is_save == True:
    #         if (epoch+1) % 100 == 0:
    #             # model_path = 'model_ae_' + str(epoch+1) + '.pth'
    #             # optim_path = 'optim_ae_' + str(epoch+1) + '.pth'
    #             model_path = 'model_ae_' + str(epoch+1) + '_s.pth'
    #             optim_path = 'optim_ae_' + str(epoch+1) + '_s.pth'
    #             torch.save(model.state_dict(), model_path)
    #             torch.save(optimizer.state_dict(), optim_path)
    
    # drawing_graph(num_epoch, train_loss_list, val_loss_list, draw_flag="loss")
    # # drawing_graph(num_epoch, train_acc_list, val_acc_list, draw_flag="accuracy")

    # # save parameters of the model
    # if is_save == True:
    #     # model_path = 'model_ae_' + str(epoch+1) + '.pth'
    #     # optim_path = 'optim_ae_' + str(epoch+1) + '.pth'
    #     model_path = 'model_ae'  + '_s.pth'
    #     optim_path = 'optim_ae' + '_s.pth'
    #     # model_path = 'model_ae.pth'
    #     # optim_path = 'optim_ae.pth'
    #     torch.save(model.state_dict(), model_path)
    #     torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    model2 = CNN_AutoEncoder().to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)   #adam  lr=0.0001
    criterion2 = torch.nn.MSELoss()
    # read parameters of the model
    # model_path = 'model_ae_50.pth'
    # model_path = 'model_ae_' + str(epoch+1) + '.pth'
    model_path = 'model_ae_s.pth'
    model2.load_state_dict(torch.load(model_path))
    # optimizer2.load_state_dict(torch.load(optim_path))

    # test
    model2.eval()
    print('Test begin...')
    ave_test_loss, outputs_and_inputs = testing(test_loader, model2, criterion2, optimizer2, device, model_flag)
    print(f"Test Loss: {ave_test_loss:.5f}")
    # 入力画像と出力画像を表示
    output_image, input_image = outputs_and_inputs[-1]
    output_image = output_image.to('cpu')
    input_image = input_image.to('cpu')
    _ = show_image(input_image.reshape(-1, 3, 256, 256), image_flag="in")
    oo = show_image(output_image.reshape(-1, 3, 256, 256), image_flag="out")

    # debug
    im_in = np.array(Image.open('movies/coco_AutoEncoder_in_sample_0702.png'))
    im_out = np.array(Image.open('movies/coco_AutoEncoder_out_sample_0702.png'))
    images = torch.Tensor(im_in)
    outputs = torch.Tensor(im_out)
    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, images)
    print(loss)

    print("o", outputs.shape)
    print("oi", output_image.shape)

    loss2 = criterion(outputs, output_image)
    print(loss2)
    

if __name__ == "__main__":
    main()