import torch
from torch.utils.data import Subset,Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import json

class RTFDataset(Dataset):
    def __init__(self, type = 'train'):
        self.root = f'data/RTF/'
        self.transform1 = transforms.Compose([transforms.ToTensor()])   # 不进行crop
        self.normal, self.abnormal = self.read_img(type)

        self.type = type
        self.nr = 0.8

        self.train_data = torch.stack(self.normal[:int(self.nr*len(self.normal))])
        self.test_data =  torch.stack(self.abnormal + self.normal[int(self.nr*len(self.normal)):]) # 归一化

        
        self.mean = torch.mean(self.train_data,(0,2,3))
        self.std = torch.mean(self.train_data,(0,2,3))
        self.transform2 = transforms.Compose([transforms.Normalize(self.mean, self.std)]) # 标准化
        # self.transform2 = transforms.Compose([transforms.Lambda(lambda x: x)])

        norm = {}
        norm['mean']= self.mean.tolist()
        norm['std']=self.std.tolist()
        norm_file = os.path.join(self.root, type, 'norm.json')
        with open(norm_file, 'w') as f:
            json.dump(norm, f)
        

    def __getitem__(self, index):
        if self.type =='train':
            image = self.transform2(self.normal[index])
            label = 0
        else:
            if index < len(self.abnormal):
                image = self.transform2(self.abnormal[index])
                label = 1
            else:
                new_index = index - len(self.abnormal)
                image = self.transform2(self.normal[-new_index])
                label = 0

        return image, label, index
        
    def __len__(self):
        if self.type == 'train':
            return int(self.nr*len(self.normal))
        else:
            return len(self.abnormal) + int((1-self.nr)*len(self.normal))

    def read_img(self, type):
        normal_images = []
        abnormal_images = []
        if type == 'train':
            img_dir = os.path.join(self.root, type, 'Non defective')
            img_list = os.listdir(img_dir)
            for img_path in img_list:
                normal_images.append(self.transform1(Image.open(os.path.join(img_dir, img_path))))
        else:
            img_dir = os.path.join(self.root, type, 'Non defective')
            img_list = os.listdir(img_dir)
            for img_path in img_list:
                normal_images.append(self.transform1(Image.open(os.path.join(img_dir, img_path))))

            img_dir = os.path.join(self.root, type, 'Defective')
            img_list = os.listdir(img_dir)
            for img_path in img_list:
                abnormal_images.append(self.transform1(Image.open(os.path.join(img_dir, img_path))))
        
        return normal_images, abnormal_images