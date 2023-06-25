import config
import torch
import torch.nn as nn

#Define the discriminator model, composed by convolutional blocks
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        #Function that initializes weights from aGaussian distribution N(0, 0.02)
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            
        self.initial = nn.Sequential(             #The first block doesn't use instance normalization
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(init_weights)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect"),    
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(init_weights)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(init_weights)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect"),   
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(init_weights)
        
        self.conv4 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect").apply(init_weights)


    def forward(self, x, feature_extract = False):
            x0 = self.initial(x)
            x1 = self.conv1(x0)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            if(feature_extract == False):
                return torch.sigmoid(self.conv4(x3))
            else:
                return x3

if __name__ == "__main__":
    test()
