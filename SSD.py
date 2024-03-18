#%%
%cd /home/tki256/myProject/pytorchによる発展ディープラーニング/pytorch_advanced/2_objectdetection

#%%
import os.path as osp
import random
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#%% 学習，検証の画像データとアノテーションデータへのファイルパスリスト作成
def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成する
    
    Parameters
    ---------------
    rootpath : str
        データフォルダへのパス
        
    Returns
    --------------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    # 訓練と検証，それぞれのファイルのID（ファイル名を取得する）
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
    
    #訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list

#%% ファイルパスのリストを作成
rootpath = "data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

print(train_img_list[0])

#%% 「XML形式のアノテーション」をリスト形式に変換するクラス
class Anno_xml2list(object):
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を画像サイズで規格化してからリスト形式に変換
    
    Attributes
    -------------------
    classes : リスト
        VOCのクラス名を格納したリスト
    """
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """
        Parameters
        --------------
        xml_path : str
            xmlファイルへのパス
        width : int
            対象画像の幅
        height : int
            対象画像の高さ
            
        Returns
        -------------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ...]
            物体のアノテーションデータを格納したリスト．画像内に存在する物体数分のだけ要素をもつ
        """
        
        #画像内の全ての物体のアノテーションをこのリストに格納
        ret = []
        
        xml = ET.parse(xml_path).getroot()
        
        for obj in xml.iter('object'):
            
            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []
            
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            
            # アノテーションのxmin, ymin, xmax, ymaxを取得し，0~1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for pt in (pts):
                # VOCは原点が(1,1)なので1を引き算して(0,0)
                cur_pixel = int(bbox.find(pt).text) - 1
                
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height
                    
                bndbox.append(cur_pixel)
            
            #アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            ret += [bndbox]
            
        return np.array(ret)
# %% 動作確認
voc_classes = ['aeroplane', 'bycycle', 'bird', 'boat', 
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

transform_anno = Anno_xml2list(voc_classes)

# 画像の読み込み openCVを使用
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path)
height, width, channels = img.shape

# アノテーションをリストで表示
transform_anno(val_anno_list[ind], width, height)

#%%
# 入力画像の前処理クラス
from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform():
    """
    画像とアノテーションの前処理クラス．訓練と推論で異なる動作をする．
    画像のサイズを300*300
    学習時はデータオーギュメンテーション
    
    Attributes
    -------------
    input_size : int
        リサイズ先の画像の大きさ
    color_mean : (B, G, R)
        各色チャネルの平均値
    """
    
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        -----------------
        phase : 'train' or 'val'
            前処理のモードを指定
        """
        return self.data_transform[phase](img, boxes, labels)

#%% 動作確認

# 1. 画像読み込み
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)
height, width, channels = img.shape

# 2. アノテーションをリストに
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)

# 3. 元画像の表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 4. 前処理クラスの作成
color_mean = (104, 117, 123)
input_size = 300
transform = DataTransform(input_size, color_mean)

# 5. train画像の表示
phase = "train"
img_transformed, boxes, labels = transform(
    img, phase, anno_list[:, :4], anno_list[:, 2])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.show()

# 6. val画像の表示
phase = "val"
img_transformed, boxes, labels = transform(
    img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.show()
 
# %% VOC2012のDataset作成

class VOCDataset(data.Dataset):
    """
    Attributes
    -------------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練か設定する
    transform : object
        前処理クラスのインスタンス
    trainsform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """
    
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        # 前処理した画像のテンソル形式のデータとアノテーションを取得
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        # 前処理した画像のテンソル形式のデータ，アノテーション，画像の高さ，幅を取得
        # 1. 画像の読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape
        
        # 2. xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)
        
        # 3. 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]
        )
        # 色チャネルの順番をBGR→RGBに変更
        # さらに（高さ，幅，色チャネル）→（色チャネル，高さ，幅）に変更
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        
        # BBoxとラベルをセットにしたnp.arrayを作成，変数名gtはground truth（答え）の略称
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, gt, height, width
    
#%% 動作確認
color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                           transform=DataTransform(
                               input_size, color_mean), transform_anno=Anno_xml2list(voc_classes)
                           )
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                         transform=DataTransform(
                             input_size, color_mean), transform_anno=Anno_xml2list(voc_classes)
                         )

# データの取り出し例
val_dataset.__getitem__(1)
        
# %% DataLoaderの作成
def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なる
    画像内の物体数が2個であれば(2,5)であるが3庫であれば(3,5)などに変化する
    この変化に対応したDataLoaderを作成するためカスタマイズしたcollate_fnを作成
    collate_fnはPytorchでリストからmini-batchを作成する関数
    ミニバッチ分の画像が並んでいるリスト変数batchに，ミニバッチ番号を指定する次元を先頭に1つ追加してリストの形を変形
    """
    
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1])) # sample[1]はアノテーションgt
        
    # imgsはミニバッチサイズのリスト
    # リストの要素はtorch.Size([3,300,300])
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換
    imgs = torch.stack(imgs, dim=0)
    
    # targetsはアノテーションデータの正解であるgtのリスト
    # リストのサイズはミニバッチサイズ
    # リストtargetsの要素は[n, 5]となる
    # nは画像ごとに異なり，画像内にある物体の数となる
    # 5は[xmin, ymin, xmax, tmax, class_index]
    
    return imgs, targets


# %% データローダーの作成

batch_size = 4

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn
)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn
)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])
images, targets = next(batch_iterator)
print(images.size())
print(len(targets))
print(targets[1].size())

# %%
print(train_dataset.__len__())
print(val_dataset.__len__())

# %% パッケージ
from math import sqrt
from itertools import product

import pandas as pd
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# %% 35層にわたる，vggモジュールを作成
def make_vgg():
    layers = []
    in_channels = 3
    
    # vggモジュールで使用する畳み込み層やマックスプーリングのチャネル数
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "MC":
            # ceilは出力サイズを計算結果に対して切り上げで整数にするモード
            # デフォルトでは出力サイズを計算結果に対して切り下げで整数にする
            # floorモード
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)

# 動作確認
vgg_test = make_vgg()
print(vgg_test)

# %% 8層にわたる，extrasモジュールを作成
def make_extras():
    layers = []
    in_channels = 1024 # vggモジュールから出力された，extraに入力される画像チャネル
    
    # extraモジュールの畳み込み層のチャネル数を設定するコンフィグレーション
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]
    
    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]
    
    return nn.ModuleList(layers)

# 動作確認
extras_test = make_extras()
print(extras_test)

# %% デフォルトボックスのオフセットを出力するloc_layers
# デフォルトボックスに対する各クラスの信頼度confidenceを出力するconf_layersを作成

def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []
    
    # VGGの22層目，conv4_3(source1)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]
    
    # VGGの最終層(source2)に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]
    
    # extra(source3)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]
    
    # extra(source4)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # extra(source5)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]
    
    # extra(source6)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

# 動作確認
loc_test, conf_test = make_loc_conf()
print(loc_test)
print(conf_test)

# %%
