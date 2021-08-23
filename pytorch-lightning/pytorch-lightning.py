#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

# add pytorch lightning library
import pytorch_lightning as pl
import comet_ml


#check if GPU is available

gpu = print(torch.cuda.get_device_name() if torch.cuda.is_available else 'cpu')

num_gpu = torch.cuda.device_count()
print('NUmber of available GPU(s):', num_gpu)

device = print('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import datasets
import torchvision.transforms as transforms

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
])


# In[11]:


ds_train = datasets.MNIST('/home/mohammadreza/share/data', train=True, download=False, transform=transform)
ds_test = datasets.MNIST('~/share/data', train=False, download=False, transform=transform)


# In[12]:


#get_ipython().system('nvidia-smi')


# In[13]:


#get_ipython().system('nvcc -V')


# In[16]:


BATCH_SIZE = 256 if num_gpu else 64
print('Number of alailable GPU: ', num_gpu)


# In[17]:


trainloader = torch.utils.data.DataLoader(ds_train, num_workers=2,
                                         batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(ds_test, batch_size=1000,
                                        shuffle=False, num_workers=2)


# In[ ]:


# defining the model with nn.Module
class MNISTModel(pl.LightningModule): # special calss allows you
    #to define many more components that are available to be trained at neural network
    def __init__(self, n=100):
        super(MNISTModel, self).__init__()
        self.l1 = nn.Linear(28 * 28, n)
        self.l2 = nn.Linear(n, 10)
        
    def forward(self, x):
        y_hat = self.l2(F.relu(self.l1(x.flatten(1))))
        
        return y_hat
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, open_epoch=True, logger=True)
        # The "self.log" method, use internal pytorch.lightning. since this
        # is for logging intermediate values
            

