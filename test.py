#%%
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

#%%
print("Pytorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)

#%% VGG-16の学習済みモデル
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval()

print(net)

#%% 入力画像の前処理クラス
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, img):
        return self.base_transform(img)

#%%
image_file_path = '1_image_classification/data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path) # [高さ][幅][色RGB]

plt.imshow(img)
plt.show()

#%%
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()

#%%
# ILSVRCのラベル情報をロードし辞意書型変数を生成します
ILSVRC_class_index = json.load(open('1_image_classification/data/imagenet_class_index.json', 'r'))
ILSVRC_class_index

#%% 出力結果からラベルを予測する後処理クラス
class ILSVRCPredictor():
    def __init__(self, class_index):
        self.class_index = class_index
        
    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]
        
        return predicted_label_name

predictor = ILSVRCPredictor(ILSVRC_class_index)

transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)
inputs = img_transformed.unsqueeze_(0)

out = net(inputs)
result = predictor.predict_max(out)

print("入力画像の予測気か：", result)

#%%
import glob
import os.path as osp
import random
import numpy as np
import json 
from  PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
