import numpy as np
import matplotlib.pyplot as plt
import torch
import sys, os 
from pycocotools.coco import COCO

anno_path = "coco/annotations/instances_train2014.json"
coco = COCO(anno_path)

# try
cat_ids = coco.getCatIds(catNms=["dog", "cat"])  # 指定したカテゴリに対応するcategory_IDを取得する
# print(coco.getCatIds(supNms=["vehicle"]))
img_ids = coco.getImgIds(catIds=cat_ids)  # 指定したカテゴリ ID の物体がすべて存在する画像の ID 一覧を取得する。

# label: 80種類
category_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# アリとハチ用のカスタムデータセット

# root = 'hymenoptera_data'
root = '../coco/images'
class Coco_Dataset(torch.utils.data.Dataset):  
    classes = ['ant', 'bee']
  
    def __init__(self, root, category_list, transform=None, data_kind="train"):
        # 指定する場合は前処理クラスを受け取る
        self.transform = transform
        # 画像とラベルの一覧を保持するリスト
        # self.images = []
        # self.labels = []
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
            # root_ants_path = os.path.join(root, 'train2014', 'ants')
            # root_bees_path = os.path.join(root, 'train2014', 'bees')
        elif data_kind == "val":
            root_path = os.path.join(root, 'val2014')
            anno_path = "coco/annotations/instances_val2014.json"
        else:
            root_path = os.path.join(root, 'test2014')
            anno_path = "coco/annotations/instances_test2014.json"

        coco = COCO(anno_path)

        # 画像一覧を取得
        all_images = os.listdir(root_path)
        for i in range(len(category_list)):
            cat_ids = coco.getCatIds(catNms=category_list[i])  # 指定したカテゴリに対応するcategory_IDを取得する
            img_ids = coco.getImgIds(catIds=cat_ids)  # 指定したカテゴリ ID の物体がすべて存在する画像の ID 一覧を取得する。

            # 指定した画像 ID に対応する画像情報とラベルを取得する。
            for j in range(len(img_ids)):
                img_info, = coco.loadImgs(img_id)
                self.images.append(os.path.join(root_path, '/', img_info["file_name"]))
                self.labels.append(category_list[i])

        # # アリの画像一覧を取得
        # ant_images = os.listdir(root_ants_path)
        # # ここではアリをラベル０に指定
        # ant_labels = [0] * len(ant_images)
        # # ハチの画像一覧を取得
        # bee_images = os.listdir(root_bees_path)
        # # ここではハチをラベル１に指定
        # bee_labels = [1] * len(bee_images)
        # 1個のリストにする
        # for image, label in zip(ant_images, ant_labels):
        #     self.images.append(os.path.join(root_ants_path, image))
        #     self.labels.append(label)
        # for image, label in zip(bee_images, bee_labels):
        #     self.images.append(os.path.join(root_bees_path, image))
        #     self.labels.append(label)
        
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
        return image, label
        
    def __len__(self):
        # ここにはデータ数を指定
        return len(self.images)