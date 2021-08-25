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


# In[2]:


import pytorch_lightning as pl
import comet_ml


# In[3]:


gpu = print(torch.cuda.get_device_name() if torch.cuda.is_available else 'cpu')


# In[16]:


AVAIL_GPUS = min(1, torch.cuda.device_count())


# In[17]:


device = print('cuda' if torch.cuda.is_available() else 'cpu')


# In[6]:


from torchvision import datasets
import torchvision.transforms as transforms

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
])


# In[7]:


ds_train = datasets.MNIST('/home/mohammadreza/share/data', train=True, download=False, transform=transform)
ds_test = datasets.MNIST('~/share/data', train=False, download=False, transform=transform)


# In[8]:


get_ipython().system('nvidia-smi')


# In[9]:


get_ipython().system('nvcc -V')


# In[10]:


BATCH_SIZE = 256 if num_gpu else 64
print('Number of alailable GPU: ', num_gpu)


# In[11]:


trainloader = torch.utils.data.DataLoader(ds_train, num_workers=2,
                                         batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(ds_test, batch_size=1000,
                                        shuffle=False, num_workers=2)


# In[21]:


# defining the model with nn.Module
class MNISTModel(pl.LightningModule): # special calss allows you
    #to define many more components that are available to be trained at neural network
    def __init__(self, n=100):
        super(MNISTModel, self).__init__()
        self.l1 = nn.Linear(28 * 28, n)
        self.l2 = nn.Linear(n, 10)
        
    def forward(self, x):
        y_hat = self.l2(F.relu(self.l1(x.flatten(1))))
        # we need to apply the layers into flatten version of input
        
        return y_hat
    
    '''
    Till here is same as the method we used is from "nn.Sequenstial" to 
    
    "nn.Module". By follosing we define "train_step", "test_step", 
    
    "validation_step" and "prediction_step" by using inner mothods of
    
    "LightningModule"
    '''
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, on_epoch=True, logger=True)
        # The "self.log" method, use internal pytorch.lightning facilities.
        # since this is for logging intermediate values
        return loss
        
    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, *args):
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001) 
        
            


# In[22]:


model2 = MNISTModel()


# In[23]:


X_batch, y_batch = iter(trainloader).next()


# In[24]:


# Draw the images
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(ds_train), size=(1,)).item()
    img, label = ds_train[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(f"{label}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Trainer

trainer = pl.Trainer(
    gpus = AVAIL_GPUS,
    max_epochs=1,
    progress_bar_refresh_rate=20,
    auto_select_gpus=True
    
)



trainer.fit(model2, trainloader) #training


# In[32]:


# if you want to validate, you can also run validation

trainer.validate(model2, testloader) #testing 

p=trainer.predict(model2, testloader) #orediction on test data
#trainer.predict(model2, trainloader) #orediction on train data


# ### Tensorboard in PytorchLightning

# In[ ]:




