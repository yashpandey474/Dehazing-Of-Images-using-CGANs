# %%
import cv2
import numpy as np
import random
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
import matplotlib.pyplot as plt
import random
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
import imageio


def add_haze(image, haze_intensity = "low"):
    # Convert image to uint8 data type
    image_uint8 = (image * 255).astype(np.uint8)

    # Simulate haze by blending the image with a white overlay
    overlay = np.full_like(image_uint8, (255, 255, 255), dtype=np.uint8)  # White overlay

    # Define range of alpha values based on haze intensity
    if haze_intensity == 'low':
        # Adjusted alpha range for low haze to make it a little more hazy
        alpha_range = (0.2, 0.5)
    elif haze_intensity == 'medium':
        alpha_range = (0.3, 0.6)
    elif haze_intensity == 'high':
        alpha_range = (0.6, 0.9)
    else:
        raise ValueError("Invalid haze intensity level. Choose from 'low', 'medium', or 'high'.")

    # Random transparency level within the specified range
    alpha = random.uniform(alpha_range[0], alpha_range[1])

    # Blend image with overlay to create haze effect
    haze_image = cv2.addWeighted(image_uint8, 1 - alpha, overlay, alpha, 0)

    return haze_image


def check_file_count(directory):
    return len(os.listdir(directory))

def augment_and_save_images(root_dir):
    # %%
    transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                                 ])
    # GET TRAIN DIRECTORY
    train_dir = os.path.join(root_dir, "train")
    # GET GROUND TRUTH DIRECTORY
    input_dir = os.path.join(train_dir, "GT")

    # GET HAZY DIRECTORY
    hazy_dir = os.path.join(train_dir, "hazy")

    if check_file_count(hazy_dir) > 20000:
        print("Your dataset is already augmented. Moving to training")
        return

    count = 0
    for clean_image_path in os.listdir(input_dir):
        # LOAD AND PASS TO FUNCTION AS IMAGE
        clean_image = Image.open(os.path.join(input_dir, clean_image_path)).convert("RGB")
        clean_image = transform(clean_image)

        clean_image = clean_image.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy array
        clean_image = clean_image * 0.5 + 0.5  # Denormalize

        # Add haze at different intensity levels
        hazy_image_low = add_haze(clean_image, 'low')
        hazy_image_medium = add_haze(clean_image, 'medium')

        # Save only low haze and medium haze images back to hazy directory
        filename = os.path.splitext(os.path.basename(clean_image_path))[0]
        Image.fromarray(hazy_image_low).save(os.path.join(hazy_dir, f'{filename} low.png'))
        Image.fromarray(hazy_image_medium).save(os.path.join(hazy_dir, f'{filename} medium.png'))

        count += 1
        print(f"Augmented {count} Image(s)")




# root_dir = '/content/drive/MyDrive/Task2/Dataset/train'

# augment_and_save_images(root_dir)

