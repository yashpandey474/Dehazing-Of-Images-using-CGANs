
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
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import models

# PATCHGAN DISCRIMINATOR
class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 2):
        super(CNNBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self,in_channels = 3, features = [64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2,features[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        ) # according to paper 64 channel doesn't contain BatchNorm2d
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels,feature,stride=1 if feature==features[-1] else 2 ))
            in_channels = feature
        
        layers.append(
            nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode='reflect')
        )
        self.model = nn.Sequential(*layers)
    
    def forward(self,x,y):
        x = torch.cat([x,y],dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
# TO PREPROCESS AND PASS TO A GENERATOR
class Model:
    def __init__(self, weights_path, transform, generator):
        self.weights_path = weights_path
        self.transform = transform
        self.generator.load_state_dict(torch.load(weights_path))
        self.generator.eval()

    # NEEDS TO BE CALLED WHEN model(image_path) IS DONE
    def apply(self, image_path):
        hazy_img = Image.open(image_path).convert('RGB')
        hazy_img = self.transform(hazy_img).unsqueeze(0)  # Add batch dimension

        dehazed_image = self.generator(hazy_img)

        dehazed_image = dehazed_image * 0.5 + 0.5

        return dehazed_image
        
    def __call__(self, image_path):
        return self.apply(image_path)
    
class Block(nn.Module):
    def __init__(self,in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super(Block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='reflect')
            if down
            else
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class GeneratorModel1(nn.Module):
    def __init__(self,in_channels=3,features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels,features,4,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features,features*2,down=True,act='leaky',use_dropout=False)
        self.down2 = Block(features*2,features*4,down=True,act='leaky',use_dropout=False)
        self.down3 = Block(features*4,features*8,down=True,act='leaky',use_dropout=False)
        self.down4 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        self.down5 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        self.down6 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode='reflect'),
            nn.ReLU()
        )

        self.up1 = Block(features*8,features*8,down=False,act='relu',use_dropout=True)
        self.up2 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up3 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up4 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=False)
        self.up5 = Block(features*8*2,features*4,down=False,act='relu',use_dropout=False)
        self.up6 = Block(features*4*2,features*2,down=False,act='relu',use_dropout=False)
        self.up7 = Block(features*2*2,features,down=False,act='relu',use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh(),
        )
    
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7],dim=1))
        up3 = self.up3(torch.cat([up2,d6],dim=1))
        up4 = self.up4(torch.cat([up3,d5],dim=1))
        up5 = self.up5(torch.cat([up4,d4],dim=1))
        up6 = self.up6(torch.cat([up5,d3],dim=1))
        up7 = self.up7(torch.cat([up6,d2],dim=1))
        return self.final_up(torch.cat([up7,d1],dim=1))



class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu'):
        super(SimpleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect') if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class GeneratorModel2(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.down1 = SimpleBlock(features, features*2, down=True)
        self.down2 = SimpleBlock(features*2, features*4, down=True)
        self.down3 = SimpleBlock(features*4, features*8, down=True)
        self.down4 = SimpleBlock(features*8, features*8, down=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )

        self.up1 = SimpleBlock(features*8, features*8, down=False)
        self.up2 = SimpleBlock(features*8*2, features*8, down=False)
        self.up3 = SimpleBlock(features*8*2, features*4, down=False)
        self.up4 = SimpleBlock(features*4*2, features*2, down=False)
        self.up5 = SimpleBlock(features*2*2, features, down=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        bottleneck = self.bottleneck(d5)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d5], dim=1))
        up3 = self.up3(torch.cat([up2, d4], dim=1))
        up4 = self.up4(torch.cat([up3, d3], dim=1))
        up5 = self.up5(torch.cat([up4, d2], dim=1))
        return self.final_up(torch.cat([up5, d1], dim=1))

def down_conv(in_channels, out_channels, kernel_size, stride, padding):
  conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2, inplace=True)
  )
  return conv

def up_conv(in_channels, out_channels, kernel_size, stride, padding):
  conv = nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
  )
  return conv

class GeneratorModel5(nn.Module):
    def __init__(self):
        super(GeneratorModel5, self).__init__()
        # Encoding layers
        self.down_conv_1 = down_conv(3,64,5,1,2)
        self.down_conv_2 = down_conv(64,128,4,2,1)
        self.down_conv_3 = down_conv(128,256,4,2,1)
        self.down_conv_4 = down_conv(256,512,4,2,1)
        self.down_conv_5 = down_conv(512,1024,4,2,1)

        self.up_trans_1 = up_conv(1024, 512, 4, 2, 1)
        self.up_conv_1 = down_conv(1024,512, 3, 1, 1)
        self.up_trans_2 = up_conv(512, 256, 4, 2, 1)
        self.up_conv_2 = down_conv(512, 256, 3, 1, 1)
        self.up_trans_3 = up_conv(256, 128, 4, 2, 1)
        self.up_conv_3 = down_conv(256, 128, 3, 1, 1)
        self.up_trans_4 = up_conv(128, 64, 4, 2, 1)
        self.up_conv_4 = down_conv(128, 64, 3, 1, 1)

        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, image):
      #encoder
                                   # image = [32, 3, 256, 256]
      x1 = self.down_conv_1(image) # x1 = [32, 64, 256, 256]
      x2 = self.down_conv_2(x1)    # x2 = [32, 128, 128, 128]
      x3 = self.down_conv_3(x2)    # x3 = [32, 256, 64, 64]
      x4 = self.down_conv_4(x3)    # x4 = [32, 512, 32, 32]
      x5 = self.down_conv_5(x4)    # x5 = [32, 1024, 16, 16]

      #decoder
      y = self.up_trans_1(x5)                   # y1 = [32, 512, 32, 32]
      y = self.up_conv_1(torch.cat([y,x4],1))  # y1 = [32, 512, 32, 32]
      y = self.up_trans_2(y)                   # y2 = [32, 256, 64, 64]
      y = self.up_conv_2(torch.cat([y,x3],1))  # y2 = [32, 256, 64, 64]
      y = self.up_trans_3(y)                   # y3 = [32, 128, 128, 128]
      y = self.up_conv_3(torch.cat([y,x2],1))  # y3 = [32, 128, 128, 128]
      y = self.up_trans_4(y)                   # y4 = [32, 64, 256, 256]
      y = self.up_conv_4(torch.cat([y,x1],1))  # y4 = [32, 64, 256, 256]
      y = self.out(y)

      return y