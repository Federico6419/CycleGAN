import torch
import torch.nn as nn

#Define the Residual Block structure, composed by a convolutional layer and an instance normalization (only the first use an activation function)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential( 
            nn.Conv2d(channels, channels, kernel_size=3, padding_mode="reflect", padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding_mode="reflect", padding=1),
            nn.InstanceNorm2d(channels)
        )
  
    def forward(self, x):
        return x + self.block(x)

#Define the generator model
class Generator(nn.Module):
    def __init__(self, img_channels, num_residuals=9):      #There are 9 Residual Blocks if the image resolution is 256, while 6 if it is 128
        super().__init__()
        #############################  CONVOLUTIONAL BLOCKS #################################
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)   
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)   
        )

        ############################   RESIDUAL BLOCKS    ##################################
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)]
        )

        ####################### TRANSPOSE CONVOLUTIONAL BLOCKS ##############    
        self.tran1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.tran2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        ####################### LAST LAYER ############## 
        self.last = nn.Conv2d(64, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.res_blocks(x)
        # x = self.tran1(x)
        x = self.tran1(x)
        x = self.tran2(x)
        
        return torch.tanh(self.last(x))
    
if __name__ == "__main__":
    test()
