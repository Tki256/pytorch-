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
        
    def __len(self):
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
        