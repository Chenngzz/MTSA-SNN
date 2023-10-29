import numpy as np
import torch
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms,datasets
import time

data_transforms = {
    'train':
        transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'valid':
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
}

class Dataset(Dataset):
    def __init__(self,data_file,image_file, transform=None):
        self.df = pd.read_csv(data_file)
        self.transform = transform
        self.image_file = image_file
        self.img = [i[0] for i in datasets.ImageFolder(self.image_file)]
        self.label = [self.df.values[i][-1] for i in range(len(self.df.values))]
        self.data  = [self.df.values[i][:-1] for i in range(len(self.df.values))]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img =self.img[idx]
        label = self.label[idx]
        data= self.data[idx]
        if self.transform :
            image =self.transform(img)
        image = torch.from_numpy(np.array(image))
        label = torch.from_numpy(np.array(label))
        data  = torch.from_numpy(np.array(data))

        tensor_fea = torch.tensor(data, dtype=torch.float16).detach()
        tensor_lab = torch.tensor(label).detach()
        tensor_img = torch.tensor(image, dtype=torch.float16).detach()

        return tensor_img, tensor_fea, tensor_lab

dataset = Dataset("/dataset/...csv", 'dataset/image/', transform=data_transforms['train'])

train_data  = int(len(dataset) * 0.8)
test_data = int(len(dataset)) - train_data


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data, test_data])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)