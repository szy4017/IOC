import torch
from torch.utils.data import Subset,Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import json

class RTFDataset(Dataset):
    def __init__(self, scene, type = 'train'):
        self.root_file = f'data/RTF/'
        self.scene = scene
        self.transform1 = transforms.Compose([transforms.Lambda(lambda x: x.crop((480,0,1920,360))),
                                              transforms.ToTensor()])
        if self.scene !='0':
            self.normal, self.abnormal = self.read_img(self.scene)
        else:
            self.normal, self.abnormal = self.read_img_all()

        self.type  = type
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
        norm_file = self.root_file + self.scene + '/norm.json'
        with open(norm_file,'w') as f:
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

    def read_img(self, scene):
        file_list = os.listdir(self.root_file + f'{scene}/normal')
        normal_images = []
        for image_path in file_list:
            normal_images.append(self.transform1(Image.open(self.root_file + f'{scene}/normal/' + image_path)))
     
        abnormal_images = []
        for image_path in os.listdir(self.root_file + f'{scene}/abnormal'):
            abnormal_images.append(self.transform1(Image.open(self.root_file + f'{scene}/abnormal/' + image_path)))
        
        return normal_images, abnormal_images

    def read_img_all(self):
        normal_images, abnormal_images = [], []
        
        for scene in range(1,21):
            # print(scene)
            normal, abnormal = self.read_img(scene)
            normal_images += normal
            abnormal_images += abnormal

        return normal_images, abnormal_images