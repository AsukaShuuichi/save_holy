import pandas as pd
import os 
import json
from sklearn.model_selection import train_test_split
#还是需要
#好像并不用进行分数据集,而且读取的数据也并非是csv
import numpy as np
from torchvision.datasets.folder import ImageFolder, default_loader
#关于这个imagefolder哈，就是它假定数据集的文件结构为每类图片放在一个子文件夹中，子文件夹的名字就是类别标签。
#自动为每个类别分配一个整数标签，并将图片与标签组成 (image, label) 的形式返回。
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class CustomDataset(Dataset):
    
    def __init__(self,root,txt_file,transform=None):
        self.root=root
        self.transform=transform
        self.samples=[]
    

        
#不对，还是不能直接拿来用

        with open(txt_file,'r') as f:
            for line in f:
                _,path,label=line.strip().split()
                self.samples.append((os.path.join(root,path),int(label)))
    def __getitem__(self,index):
        path,label=self.samples[index]
        image=Image.open(path).convert('RGB')

        if self.transform:
            image=self.transform(image)
        return {"image": image, "label": label}
    #用于定义如何根据索引返回数据集中的一项
    def __len__(self):
        return len(self.samples)