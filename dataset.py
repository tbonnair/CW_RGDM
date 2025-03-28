import torch
import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
import torchvision

class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        assert os.path.exists(root_dir) == True
        for subdir, _, files in os.walk(root_dir):
            # print("files", files)
            for file in files:
                if file.endswith(".png"):
                    self.image_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Added TB 250318
class Phi4Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        labels = 1
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, labels
    

def get_dataset(dataset_key):

    if dataset_key == "cifar10":
        dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    
    elif dataset_key == "ffhq32":
        dataset = FFHQDataset(
            root_dir="./data/ffhq_dataset/ffhq32",            
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    elif dataset_key == "ffhq64":
        dataset = FFHQDataset(
            root_dir="./data/ffhq_dataset/ffhq64",            
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    elif dataset_key == 'phi4':
        filename = 'phi4_L128_Beta0.68'     #'phi4_L32'     #'phi4_L128_Beta0.68'
        data = torch.tensor(np.load('./data/Phi4/{:s}.npy'.format(filename))).unsqueeze(1).float()
        rotation = RandomRotationTransform(angles=[0, 90, 180, 270])  
        mytrans = transforms.Compose([
            rotation,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize((0.,), (2,))
        ])
        dataset = Phi4Dataset(data, mytrans)
        
    elif dataset_key == 'CW':
        filename = '/home/bonnaire/data/256/'
        shape = (32, 32)
        rotation = RandomRotationTransform(angles=[0, 90, 180, 270])  
        mytrans = transforms.Compose([
            torchvision.transforms.RandomCrop(size=shape),
            rotation,
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize((0,), (2,))
        ])
        dataset = get_dataset_CW(shape,
                                 filename,
                                 mytrans)
    else:
        dataset = None
    
    return dataset


def get_dataset_CW(shape=(16, 16), path='./', transform=None):
    densities = []
    for i in range(64):
        dataset = torch.load(path + 'ldensity_{:d}.pt'.format(i))
        densities.append(dataset)
        
    dataset = torch.vstack(densities)[::2]
    dataset = dataset[:, None, :, :]
    dataset = Phi4Dataset(dataset, transform)
    return dataset
