import os
from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from PIL import Image
import torch
import torch.utils.data as data 
import torchvision
import torchvision.transforms as transforms


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    return dataset


def load_hsr(scene):
    
    root_file = f'hsr/{scene}'
    file_list = os.listdir(root_file)
    file_list.sort()

    transform1 = transforms.Compose([transforms.ToTensor()])
    normal_images = []
    for image_path in file_list:
        if image_path.split('.')[-1]=='jpg':
            normal_images.append(transform1(Image.open(root_file + '/' + image_path)))
    
    abnormal_images = []
    for image_path in os.listdir(root_file + '/abnormal'):
        abnormal_images.append(transform1(Image.open(root_file + '/abnormal/' + image_path)))
    
    normal_dataset = data.TensorDataset(normal_images, torch.zeros(len(normal_images)))
    abnormal_dataset = data.TensorDatasetDataset(abnormal_images, torch.zeros(len(abnormal_images)))
    
    train_set = normal_dataset
    test_set = abnormal_dataset

    return train_set, test_set



