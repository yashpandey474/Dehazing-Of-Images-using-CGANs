
TEST_DATASET_HAZY_PATH = 'C:/viki/dl/Task_2/t2dataset/test/hazy'
TEST_DATASET_OUTPUT_PATH = 'C:/viki/dl/Task_2/t2dataset/test/generated'

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
from model_architectures import GeneratorModel1
import numpy as np

input_images = os.listdir(TEST_DATASET_HAZY_PATH)
num = len(input_images)
output_images = []

for i in range(num):
    output_images.append(os.path.join(TEST_DATASET_OUTPUT_PATH, input_images[i]))
    input_images[i] = os.path.join(TEST_DATASET_HAZY_PATH, input_images[i])

'''
Write the code here to load your model
'''
weights_path = 'generator_l1_cgan.pth'
model = GeneratorModel1()
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

        
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    hazy_img = Image.open(image_path).convert('RGB')
    preprocessed_image = transform(hazy_img).unsqueeze(0)  # Add batch dimension
    
    return preprocessed_image

def save_generated_image(dehazed_image, output_path):

    dehazed_image = dehazed_image.permute(1, 2, 0)
    dehazed_image = dehazed_image * 0.5 + 0.5    

    dehazed_image = (dehazed_image.detach().cpu().numpy()*255).astype(np.uint8)
    
    # SAVE THE IMAGE
    # Convert the numpy array to PIL Image
    single_image_pil = Image.fromarray(dehazed_image)

    # Save the image using PIL
    single_image_pil.save(output_path)

# MAKE OUTPUT DIRECTORY IF DOES NOT EXIST
os.makedirs(TEST_DATASET_OUTPUT_PATH, exist_ok=True)

for i in range(num):
    # PREPROCESS IMAGE
    preprocessed_image = preprocess_image(input_images[i])

    # PASS PREPROCESSED IMAGE TO MODEL
    dehazed_img = model(preprocessed_image)
   
    dehazed_img = dehazed_img.squeeze()

    # now save the dehazed image at the path indicated by output_images[i]
    save_generated_image(dehazed_img, output_images[i])
