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

class DehazingDataset1(data.Dataset):
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
    
class DehazingDataset(data.Dataset):
    def _init_(self, root_dir, transform=None):
        #Get the images
        self.root_dir = root_dir
        hazy_images_path = os.path.join(root_dir, 'hazy')
        clean_images_path = os.path.join(root_dir, 'GT')


        hazy_images = [f for f in os.listdir(hazy_images_path) if  f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
      
        self.hazy_images = []
        self.clean_images = []

        for path in hazy_images:
            basename = os.path.splitext(os.path.basename(path))[0]  # Extract base filename without extension
            filename = basename.split()[0]
             
            clean_image = os.path.join(clean_images_path, filename + '.png')  # Assuming clean images have PNG extension
            if not os.path.exists(clean_image):
                clean_image = os.path.join(clean_images_path, filename + '.jpg')  # Try JPG extension
            if not os.path.exists(clean_image):
                clean_image = os.path.join(clean_images_path, filename + '.jpeg')  # Try JPEG extension
            if os.path.exists(clean_image):
                self.hazy_images.append(os.path.join(hazy_images_path, path))
                self.clean_images.append(clean_image)


        self.size = len(self.hazy_images)
        self.transform=transform

    def _getitem_(self, index):
        hazy_img = self.rgb_loader(self.hazy_images[index])
        clean_img = self.rgb_loader(self.clean_images[index])
        hazy_img = self.transform(hazy_img)
        clean_img = self.transform(clean_img)
        return hazy_img, clean_img

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _len_(self):
        return self.size
