from operator import index
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype('uint8')
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transforms(image)

        return{
            'image': image,
            'label': label,
            'index': index
        } 
    
class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype('uint8')
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transforms(image)

        return{
            'image': image,
            'label': label,
            'index': index
        } 