import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

class DehazingDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        hazy_images_path = os.path.join(root_dir, 'hazy')
        clean_images_path = os.path.join(root_dir, 'GT')

        self.image_pairs = []
        self.transform = transform if transform else transforms.ToTensor()

        hazy_files = sorted(os.listdir(hazy_images_path))
        clean_files = sorted(os.listdir(clean_images_path))

        self.hazy_files = hazy_files
        self.clean_files = clean_files

        count = 0
        for hazy_file, clean_file in zip(hazy_files, clean_files):
            print("Loading image pair number: ", count)
            count += 1
            
            if os.path.splitext(hazy_file)[0] == os.path.splitext(clean_file)[0]:
                hazy_path = os.path.join(hazy_images_path, hazy_file)
                clean_path = os.path.join(clean_images_path, clean_file)
                self.image_pairs.append((self.transform(self.rgb_loader(hazy_path)), self.transform(self.rgb_loader(clean_path))))

    def __getitem__(self, index):
        hazy_img, clean_img = self.image_pairs[index]
        return hazy_img, clean_img

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.image_pairs)
