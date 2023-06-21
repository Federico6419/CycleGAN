#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import config
import torch
import torch.nn as nn

#define the discriminatore model, composed by convolutional block
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial = nn.Sequential(             #The first block don t use instance normalization
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect"),    
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect"),   
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # layer if use more features, obivoiusly change parameters and name of the last layer
        """
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect"),   
            nn.InstanceNorm2d(1024),      #Invece di batch norm usa instance norm
            nn.LeakyReLU(0.2, inplace=True),
        )
        
       """
        
        self.conv4 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect")


    def forward(self, x, feature_extract = False):
            x0 = self.initial(x)
            x1 = self.conv1(x0)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            # x4 = self.conv4(x3)
            if(feature_extract == False):
                if(config.BCE or config.WAS):
                    return self.conv4(x3)
                else:
                    return torch.sigmoid(self.conv4(x3))
            else:
                return x3

if __name__ == "__main__":
    test()
