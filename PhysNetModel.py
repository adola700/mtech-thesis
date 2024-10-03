import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------------------------------------------------
# PhysNet model
# 
# the output is an ST-rPPG block rather than a rPPG signal.
# -------------------------------------------------------------------------------------------------------------------
class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3):
        super().__init__()

        self.S = S # S is the spatial dimension of ST-rPPG block

        # self.pre_start = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(1, 1, 1), stride=1, padding=(0,0,0))
            
        # my_weight = torch.Tensor([[0.299,0.587,0.114], [-0.169, -0.331, 0.5], [0.5,-0.49, -0.081]]).reshape((3,3,1,1,1))

        # self.pre_start.weight = nn.Parameter(my_weight)
        # self.pre_start.bias = nn.Parameter(torch.Tensor([0.0,128.0,128.0]))

        # # Freeze the layer by setting requires_grad to False
        # for param in self.pre_start.parameters():
        #     param.requires_grad = False

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ELU()
        )
                    

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=256, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
            
        )

        # encoding layers
        self.conv1 = nn.Conv1d(2, 2, 3, stride=1,padding = "same")
        # self.conv2 = nn.Conv1d(16, 32, 3, stride=1)
        # self.conv3 = nn.Conv1d(32, 64, 3, stride=1)
        # self.conv4 = nn.Conv1d(64, 96, 3, stride=1)
        # self.conv5 = nn.Conv1d(96, 128, 3, stride=1)
        # self.conv6 = nn.Conv1d(128, 156, 3, stride=1)
        

        # decoding layers
        # self.deconv6 = nn.ConvTranspose1d(156, 128, kernel_size=3, stride=1)
        # self.deconv5 = nn.ConvTranspose1d(128, 96, kernel_size=3, stride=1)
        # self.deconv4 = nn.ConvTranspose1d(96, 64, kernel_size=3, stride=1)
        # self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1)
        # self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1)
        # self.deconv1 = nn.ConvTranspose1d(16, 2, kernel_size=3, stride=1)
        
        # activations
        self.activ1 = nn.PReLU() #not inplace, I want to copy
        self.activ2 = nn.Tanh()

    def forward(self, x):
 
        parity = []
        
        # x = self.pre_start(x)
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds # (B, C, T, 128, 128)
        
        x = self.start(x) # out shape - (B, 32, T, 128, 128)
        x = self.loop1(x) # (B, 64, T, 64, 64)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x) # (B, 64, T/2, 32, 32)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x) # (B, 64, T/4, 16, 16)
        x = self.loop4(x) # (B, 64, T/4, 8, 8)

        x = F.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T/2, 8, 8)
        x = self.decoder1(x) # (B, 64, T/2, 8, 8)
        x = F.pad(x, (0,0,0,0,0,parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T, 8, 8)
        x = self.decoder2(x) # (B, 64, T, 8, 8)
        x = F.pad(x, (0,0,0,0,0,parity[-2]), mode='replicate')
        x = self.end(x) # (B, 1, T, S, S), ST-rPPG block

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:,:,:,a,b]) # (B, 1, T)


        x = sum(x_list)/(self.S*self.S) # (B, 1, T)
        X = torch.cat(x_list+[x], 1) # (B, M, T), flatten all spatial signals to the second dimension
        # X_row1 = X[:,:2,:]
        # X_row2 = X[:,2:4,:]

        # encoder
        # x = self.conv1(X_row1)
        # res1 = self.activ1(x)
        
        # x = self.conv2(res1)
        # res2 = self.activ1(x)
        
        # x = self.conv3(res2)
        # x = self.activ1(x)
        
        # x = self.conv4(res3)
        # res4 = self.activ1(x)
        
        # x = self.conv5(res4)
        # res5 = self.activ1(x)
        
        # x = self.conv6(res5)
        # x = self.activ1(x)   
        
        
        
        # # decoder
        # x = self.deconv6(x)
        # x = self.activ1(x)
        # x += res5
        
        # x = self.deconv5(x)
        # x = self.activ1(x)
        # x += res4
        
        # x = self.deconv4(x)
        # x = self.activ1(x)
        # x += res3
        
        # x = self.deconv3(x)
        # x = self.activ1(x)
        # x += res2
        
        # x = self.deconv2(x)
        # x = self.activ1(x)
        # x += res1
        
        # x1 = self.deconv1(x)
        # # BLOCK 2

        # x = self.conv1(X_row2)
        # res1 = self.activ1(x)
        
        # x = self.conv2(res1)
        # res2 = self.activ1(x)
        
        # x = self.conv3(res2)
        # x = self.activ1(x)
        
        # # x = self.conv4(res3)
        # # res4 = self.activ1(x)
        
        # # x = self.conv5(res4)
        # # res5 = self.activ1(x)
        
        # # x = self.conv6(res5)
        # # x = self.activ1(x)   
        
        
        
        # # # decoder
        # # x = self.deconv6(x)
        # # x = self.activ1(x)
        # # x += res5
        
        # # x = self.deconv5(x)
        # # x = self.activ1(x)
        # # x += res4
        
        # # x = self.deconv4(x)
        # # x = self.activ1(x)
        # # x += res3
        
        # x = self.deconv3(x)
        # x = self.activ1(x)
        # x += res2
        
        # x = self.deconv2(x)
        # x = self.activ1(x)
        # x += res1
        
        # x2 = self.deconv1(x)
        # # x2 = self.activ1(x)

        # concatenated_tensor = torch.cat((x1, x2), dim=1)
        # # x = sum(x_list)/(self.S*self.S) # (B, 1, T)
        # shapes = concatenated_tensor.shape
        # mean_tensor = torch.mean(concatenated_tensor, dim=1).view(shapes[0],1,shapes[2])
        # # X = torch.cat(x_list+[x], 1) # (B, M, T), flat
        # final_tensor = torch.cat((concatenated_tensor, mean_tensor), dim=1)

        return X
