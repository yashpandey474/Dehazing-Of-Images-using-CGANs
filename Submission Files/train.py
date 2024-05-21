# %% [markdown]
# # GOAL - WRITE MODULAR CODE TO TRAIN MODELS
# 1. Different transforms for clean image, hazy image, test images
# 2. Training using a function with batch size etc as arguments

# %%
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
from model_architectures import GeneratorModel1, GeneratorModel2, Discriminator
from data_augmentation import augment_and_save_images
from utils import *



# %%


class Trainer:
    def __init__(self, generator, discriminator, train_dataloader, lr_step_size, lr_gamma, lambda_adv=1, lambda_res=150,
                 lambda_per=150, lambda_reg = 0.00001, num_epochs=10, wgan=False, n_critic=1, use_l1_loss=True, use_adversarial_loss=True, use_perceptual_loss = True):
        self.generator = generator
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.optimizer_G = optim.RMSprop(generator.parameters(), lr=0.00005) if wgan else optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.00005) if wgan else optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.scheduler_G = lr_scheduler.StepLR(self.optimizer_G, lr_step_size, lr_gamma)
        self.scheduler_D = lr_scheduler.StepLR(self.optimizer_D, lr_step_size, lr_gamma)
        self.criterion_G = nn.BCEWithLogitsLoss()
        self.criterion_D = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.n_critic = n_critic if wgan else 1
        self.use_l1_loss = use_l1_loss
        self.use_adversarial_loss = use_adversarial_loss
        self.use_perceptual_loss = use_perceptual_loss
        self.is_wgan = wgan
        self.perceptual_loss_net = models.vgg19(pretrained=True).features[:18].eval()
        self.lambda_adv = lambda_adv
        self.lambda_res = lambda_res
        self.lambda_per = lambda_per
        self.lambda_reg = lambda_reg
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_summary(self):
        print("Optimizer Summary:")
        print(f"Generator Optimizer: {self.optimizer_G}")
        print(f"Discriminator Optimizer: {self.optimizer_D}")
        print("Scheduler Summary:")
        print(f"Generator Scheduler: {self.scheduler_G}")
        print(f"Discriminator Scheduler: {self.scheduler_D}")
        print("Criterion Summary:")
        print(f"Generator Criterion: {self.criterion_G}")
        print(f"Discriminator Criterion: {self.criterion_D}")
        print("Other Parameters:")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Number of Critic Updates per Generator Update: {self.n_critic}")
        print(f"Use L1 Loss: {self.use_l1_loss}")
        print(f"Use Adversarial Loss: {self.use_adversarial_loss}")
        print(f"Use Perceptual Loss: {self.use_perceptual_loss}")
        print(f"Is WGAN: {self.is_wgan}")
        print(f"Device: {self.device}")

    def train(self):
        self.print_summary()
        epochs = []
        g_losses = []
        d_losses = []
        for epoch in range(self.num_epochs):
            self.scheduler_G.step()
            self.scheduler_D.step()

            g_total_loss = 0
            d_total_loss = 0
            batch_no = 1

            for hazy_imgs, clean_imgs in tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{self.num_epochs}'):
                # NON-WGAN TRAINING
                if not self.is_wgan:
                    g_complete_loss, d_loss, fake_imgs = self.train_step_nonwgan(hazy_imgs, clean_imgs)
                    
                # WGAN TRAINING
                elif self.is_wgan:
                    g_complete_loss, d_loss, fake_imgs = self.train_step_wgan(hazy_imgs, clean_imgs)

                g_total_loss += g_complete_loss.item()
                d_total_loss += d_loss.item()

                epochs.append(epoch + batch_no/len(self.train_dataloader))
                if self.is_wgan:
                    g_losses.append(-g_complete_loss.item())  # Negative because the loss is actually maximized in WGAN.
                    d_losses.append(-d_loss.item())
                else:
                    g_losses.append(g_complete_loss.item())
                    d_losses.append(d_loss.item())


                if batch_no % 10 == 0:
                    self.show_images(hazy_imgs, clean_imgs, fake_imgs, num_images=5)


                if batch_no % 20 == 0:
                    self.plot_losses(epochs, g_losses, d_losses)

                batch_no += 1

            g_avg_loss = g_total_loss / len(self.train_dataloader)
            d_avg_loss = d_total_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Generator Avg. Loss: {g_avg_loss:.4f}, Discriminator Avg. Loss: {d_avg_loss:.4f}")

            if (epoch + 1) % 3 == 0:
                self.save_samples(epoch)

        self.save_models()

    def plot_losses(self, epochs, g_losses, d_losses):
        plt.plot(epochs, g_losses, label='Generator Loss')
        plt.plot(epochs, d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs. Losses')
        plt.legend()
        plt.grid(True)
        plt.show()

    def train_step_wgan(self, hazy_imgs, clean_imgs):
        real_imgs = clean_imgs
        fake_imgs = self.generator(hazy_imgs)
                    
        for discr_train in range(self.n_critic):
            real_outputs = self.discriminator(hazy_imgs, real_imgs)
            fake_outputs = self.discriminator(hazy_imgs, fake_imgs.detach())

                        
            # UPDATE THE DISCRIMINATOR [CRITIC]
            self.optimizer_D.zero_grad()
                
            # WGAN utility, we ascend on this hence the loss will be the negative.
            d_loss = -torch.mean(real_outputs - fake_outputs)
                
            d_loss.backward()
            self.optimizer_D.step()
                
            # CLIPPING OF THE DISCRIMINATOR WEIGHTS
            for param in self.discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)
            
        # UPDATE THE GENERATOR
        self.optimizer_G.zero_grad()
            
        # REGENERATE IMAGES AND GET OUTPUTS FROM DISCRIMINATOR
        fake_imgs = self.generator(hazy_imgs)
        fake_outputs = self.discriminator(hazy_imgs, fake_imgs)
            
        #  W-LOSS FOR GENERATOR
        g_loss = -torch.mean(fake_outputs)
        g_loss.backward()
                    
        self.optimizer_G.step()

        return g_loss, d_loss, fake_imgs
    

    def train_step_nonwgan(self, hazy_imgs, clean_imgs):
        # d_loss, g_loss = self.train_step(hazy_imgs.to(self.device), clean_imgs.to(self.device))
        self.optimizer_D.zero_grad()
        real_imgs = clean_imgs
    
        # GENERATOR TAKES HAZY IMAGES AS INPUT
        fake_imgs = self.generator(hazy_imgs)
            
        # PREDICTIONS OF DISCRIMINATOR FOR REAL IMAGES
        real_outputs = self.discriminator(hazy_imgs, real_imgs)
            
        # PREDICTIONS OF DISCRIMINATOR FOR FAKE IMAGES
        fake_outputs = self.discriminator(hazy_imgs, fake_imgs.detach())
            
        # CREATE LABELS FOR LOSS CALCULATION
        real_labels = torch.ones_like(real_outputs)
        fake_labels = torch.zeros_like(fake_outputs)
            
        d_loss_real = self.criterion_D(real_outputs, real_labels)
        d_loss_fake = self.criterion_D(fake_outputs, fake_labels)
        d_loss = (d_loss_real + d_loss_fake)/2
            
        # Update discriminator
        d_loss.backward()
        self.optimizer_D.step()
            
        # Training the generator
        self.optimizer_G.zero_grad()
        fake_imgs = self.generator(hazy_imgs)
        fake_outputs = self.discriminator(hazy_imgs, fake_imgs)
        g_loss = self.criterion_G(fake_outputs, real_labels)
            
        # Compute reconstruction loss
        g_res_loss = 0
        if self.use_l1_loss:
            g_res_loss = self.l1loss(fake_imgs, clean_imgs)


        g_reg_loss = self.lambda_reg * (
            torch.sum(torch.abs(fake_imgs[:, :, :-1, :] - fake_imgs[:, :, 1:, :])) +  # Along height
            torch.sum(torch.abs(fake_imgs[:, :, :, :-1] - fake_imgs[:, :, :, 1:]))  # Along width
        )

        g_l2_loss = self.l2loss(fake_imgs, clean_imgs)
            
        # Update generator
        g_complete_loss = (self.lambda_adv *  g_loss + self.lambda_res * g_res_loss + self.lambda_per * g_l2_loss + g_reg_loss)
        g_complete_loss.backward()
        self.optimizer_G.step()

        return g_complete_loss, d_loss, fake_imgs

    def clamp_discriminator_parameters(self):
        for param in self.discriminator.parameters():
            param.data.clamp_(-0.01, 0.01)


    def save_samples(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            for i, (hazy_imgs, clean_imgs) in enumerate(self.train_dataloader):
                fake_imgs = self.generator(hazy_imgs)
                fake_imgs = fake_imgs * 0.5 + 0.5
                save_image(fake_imgs, f"sample_{epoch}_batch_{i}.png")
        self.generator.train()


    def show_images(self, hazy_imgs, clean_imgs, generated_imgs, num_images=5):
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

    def save_models(self):
        torch.save(self.generator.state_dict(), 'generator.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator.pth')






# CREATE THE DATALOADERS: CHANGE THIS PATH TO PATH WHERE DIRECTORY CONTAING TRAIN AND VAL FOLDERS IS LOCATED
root_dir = 'Dataset/'

# AUGMENT THE DATASET
print("Augmenting and saving Images: ")
augment_and_save_images(root_dir)


# GET TRAIN AND VAL DATALOADERS
train_dataloader, val_dataloader = create_train_val_dataloaders(root_dir)


# Instantiate Generator and Discriminator

# UNCOMMENT THIS IF YOU WISH TO TEST MODEL - 5 [HIGHEST PERFORMING BUT HEAVY MODEL]
generator
generator = GeneratorModel1()
discriminator = Discriminator()

# DEFINE HYPERPARAMETERS
lr_step_size = 2
lr_gamma = 0.5

# INITIALISE TRAINER AND START TRAINING
trainer = Trainer(generator, discriminator, train_dataloader, lr_step_size, lr_gamma, wgan = False)
trainer.train()

