#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import datetime
from torchsummary import summary
import torchvision.utils as vutils
import numpy as np


# In[ ]:


#选择gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)
latent_size = 64
hidden_size = 256
image_size = 28**2
num_epochs = 300
batch_size = 128
lr = 0.0002
sample_dir = 'my_samples3'


# In[ ]:


# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[ ]:


transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) # 3 for RGB channels

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../',
                                   train=True,
                                   transform=transform,
                                   download=False)
mnist_test = torchvision.datasets.MNIST(root='../',
                                   train=False,
                                   transform=transform,
                                   download=False)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=batch_size, 
                                               shuffle=False,
                                               pin_memory=True)




# In[ ]:


class Discriminator(nn.Module):
    def __init__(self,*k,**k1):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            #1 * 28 * 28
            nn.Conv2d(1,4,5), # 4 * 24 * 24
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(3), # 4 * 8 * 8
            nn.Conv2d(4,16,5,padding=1,stride=2), # 16 * 3 * 3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3) # 16 * 1 * 1
        )
        self.fc = nn.Sequential(
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.reshape(x.shape[0],1,28,28)
        y = self.conv(x)
        y = y.view(y.shape[0],-1)
        y = self.fc(y)
        return y


# In[ ]:


feature_channel = 16
class Generator(nn.Module):
    def __init__(self,*k,**k1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size,feature_channel*4,4,bias=False), # 64 * 4 * 4
            nn.BatchNorm2d(feature_channel*4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_channel*4,feature_channel*2,7,2,1,bias=False),  # 32  * 11 * 11
            nn.BatchNorm2d(feature_channel*2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_channel*2,feature_channel,7,2,1,bias=False),  # 16  * 25 * 25
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_channel,1,4,bias=False),  # 1  * 28 * 28
            nn.Tanh(),  #激活函数
        )
    def forward(self, x):
        y = self.main(x)
        return y


# In[ ]:


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator(latent_size,hidden_size,28**2)
netD = Discriminator(28**2,256,1)
netG.apply(weights_init)
netD.apply(weights_init)
netG.to(device)
netD.to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adadelta(netD.parameters())
optimizerG = optim.Adadelta(netG.parameters())
# optimizerD = torch.optim.Adam(netD.parameters(), lr=lr)
# optimizerG = torch.optim.Adam(netG.parameters(), lr=lr)

# netG.cuda()
summary(netG,input_size=(latent_size,1,1))
summary(netD,input_size=(1,28,28))


# In[ ]:



def train(netG,netD,num_epochs,optG,optD,data_loader,test_data_loader,criterion):
    
    for one in data_loader:
        temp = denorm(one[0])
        save_image(temp,os.path.join(sample_dir,'real.png'))
        break
    for epoch in range(num_epochs):
        with torch.no_grad():
            fake_images = netG(torch.randn((batch_size,latent_size,1,1),device=device))
            fake_images = fake_images.reshape(fake_images.shape[0],1,28,28)
            fake_images = denorm(fake_images)
            save_image(fake_images,os.path.join(sample_dir,'fake_images-{}.png'.format(epoch)))
            
        for i,data in enumerate(data_loader,0):
            netD.zero_grad()
            
            
            real_pic = data[0].to(device)
            b_size = real_pic.size(0)
            
            noise = torch.randn((b_size,latent_size,1,1),device=device)
            fake_pic = netG(noise)
            
            real_label = torch.full((b_size,),1,dtype=torch.float,device=device)
            fake_label = torch.full((b_size,),0,dtype=torch.float,device=device)
            
            output1 = netD(real_pic).view(-1)
            output2 = netD(fake_pic.detach()).view(-1)
       #     print(output.shape,label.shape)
            errD_real = criterion(output1,real_label)
            errD_fake = criterion(output2,fake_label)
            errD_sum = errD_real + errD_fake
            errD_sum.backward()
            
            D_x = output1.mean().item()
            D_G_z1 = output2.mean().item()
            
            optD.step()

            netG.zero_grad()
            output = netD(fake_pic).view(-1)
            errG = criterion(output,real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optG.step()
            if i % 600 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(data_loader),
                     errD_sum.item(), errG.item(), D_x, D_G_z1, D_G_z2))


# In[ ]:


start = datetime.datetime.now()

train(netG,netD,num_epochs,optimizerG,optimizerD,data_loader,test_data_loader,criterion)

print(datetime.datetime.now()-start)

