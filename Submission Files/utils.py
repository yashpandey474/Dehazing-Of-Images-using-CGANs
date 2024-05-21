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
from torch.optim import lr_scheduler
from dataset import DehazingDataset
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import models

transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# FUNCTIONS TO CREATE DATALOADERS
def create_dataloader(directory, batch_size=32, mean=0.5, std=0.5, transform = None):
    dataset = DehazingDataset(directory, transform = transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

def create_train_val_dataloaders(root_dir, train_batch_size=32, val_batch_size=32, mean=0.5, std=0.5):
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    train_dataloader = create_dataloader(train_dir, batch_size=train_batch_size, mean=mean, std=std, transform = transform)
    val_dataloader = create_dataloader(val_dir, batch_size=val_batch_size, mean=mean, std=std, transform = transform)

    return train_dataloader, val_dataloader


def show_one_images(clean_imgs, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 10))
    for i in range(num_images):
        clean_image = clean_imgs[i].detach().permute(1, 2, 0).cpu().numpy()
        clean_image = clean_image * 0.5 + 0.5

        # Plot clean images
        axes[i].imshow(clean_image)
        axes[i].axis('off')
        axes[i].set_title("Clean Image")

    plt.show()
def show_images(hazy_imgs, clean_imgs, generated_imgs, num_images=5):
    fig, axes = plt.subplots(3, num_images, figsize=(15, 10))
    for i in range(num_images):
        clean_image = clean_imgs[i].detach().permute(1, 2, 0).cpu().numpy()
        hazy_image = hazy_imgs[i].detach().permute(1, 2, 0).cpu().numpy()
        generated_image = generated_imgs[i].detach().permute(1, 2, 0).cpu().numpy()
    

        clean_image = clean_image * 0.5 + 0.5
        hazy_image = hazy_image * 0.5 + 0.5
        generated_image = generated_image * 0.5 + 0.5

        
        # Plot hazy images
        axes[0, i].imshow(hazy_image)
        axes[0, i].axis('off')
        axes[0, i].set_title("Hazy Image")

        # Plot clean images
        axes[1, i].imshow(clean_image)
        axes[1, i].axis('off')
        axes[1, i].set_title("Clean Image")

        # Plot generated images
        axes[2, i].imshow(generated_image)
        axes[2, i].axis('off')
        axes[2, i].set_title("Generated Image")

    plt.tight_layout()
    plt.show()

def show_images_save(dataloader, num_images=5, save_path = None):
    # Get a batch of data
    data_iter = iter(dataloader)
    _, images = next(data_iter)

    # Plot original clean images
    fig, axes = plt.subplots(4, num_images, figsize=(15, 15))
    for i in range(num_images):
        clean_image = images[i].permute(1, 2, 0).cpu().numpy()  # Convert to NumPy array
        # clean_image = clean_image * 0.5 + 0.5  # Denormalize
        clean_image = clean_image * 0.5 + 0.5  # Denormalize and convert to uint8


        axes[0, i].imshow(clean_image)
        axes[0, i].axis('off')
        axes[0, i].set_title("Clean Image")

        # Add haze at different intensity levels
        hazy_image_low = add_haze(clean_image, 'low')
        axes[1, i].imshow(hazy_image_low)
        axes[1, i].axis('off')
        axes[1, i].set_title("Low Haze")

        hazy_image_medium = add_haze(clean_image, 'medium')
        axes[2, i].imshow(hazy_image_medium)
        axes[2, i].axis('off')
        axes[2, i].set_title("Medium Haze")

        hazy_image_high = add_haze(clean_image, 'high')
        axes[3, i].imshow(hazy_image_high)
        axes[3, i].axis('off')
        axes[3, i].set_title("High Haze")


    if save_path:
      # Image.fromarray(clean_image).save('clean_image_{i}.png')
      Image.fromarray(hazy_image_low).save('hazy_image_low_{i}.png')
      Image.fromarray(hazy_image_medium).save('hazy_image_medium_{i}.png')
      Image.fromarray(hazy_image_high).save('hazy_image_high_{i}.png')



    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path)

    plt.show()

# %% [markdown]
# # Load Images into dataloader
