#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# In[2]:


from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


# In[3]:


ds_train = datasets.CIFAR10("~/share/data/", train=True, download=False, transform=transform)
ds_test = datasets.CIFAR10("~/share/data/", train=False, download=False, transform=transform)


# In[4]:


trainloader = torch.utils.data.DataLoader(ds_train, batch_size=512,
                                         num_workers=2, shuffle=True)


# In[5]:


testloader = torch.utils.data.DataLoader(ds_test, batch_size=512, 
                                        num_workers=2, shuffle=False)


# In[6]:


batch_X, batch_y = iter(trainloader).next()


# In[7]:


batch_X.size()


# In[8]:


import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5), dpi=150)
plt.axis('off')

plt.imshow(
    torch.transpose(
        torch.cat(
            [batch_X[batch_y==c][:30] for c in range(30)], axis=0
        ).reshape(30, 30, 32, 32),
        1, 2
    ).reshape(960, 960)
)
plt.show()


# In[9]:


device = print('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[10]:


batch_y.size()


# In[11]:


model = nn.Sequential(
    nn.Linear(3072, 500),
    nn.ReLU(),
    #nn.Linear(2500, 1000),
    #nn.ReLU(),
    #nn.Linear(1000, 100),
    #nn.ReLU(),
    nn.Linear(500, 10),
).to(device)

for w in model.parameters():
    print(' ', w.size())


# In[12]:


def flatten_trailing(batch):
    flat = batch.flatten(1)
    return flat


# In[13]:


prediction = model(flatten_trailing(batch_X.to(device)))


# In[14]:


loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(prediction, batch_y.to(device))
loss


# In[15]:


for i in range(3):
    print(model[i])


# In[16]:


print('Before backward \n', model[0].weight.grad)


# In[17]:


#loss.backward()
#print('After gradient \n', model[0].weight.grad)


# In[18]:


import tqdm
from IPython.display import clear_output 


# In[19]:


def train_loop(model, num_epochs=1, batch_size=512, loss_function = loss_fn, device='cpu'):
    train_losses = []
    test_losses = []
    test_accuracy = []
    model_div = model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_fun = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        t = tqdm.tqdm(iter(trainloader), leave=False, total=len(trainloader))
        for idx, data in enumerate(t):
            X_batch, y_batch = map(lambda x: x.to(device), data)
            #X_batch = torch.tensor(X, requires_grad=True, dtype=torch.float64)
            #y_batch = torch.tensor(y, requires_grad=True, dtype=torch.float64)
            
            loss = loss_fn(model(flatten_trailing(X_batch)), y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
            train_losses.append(loss.item())
        
        test_X, test_y = map(lambda x: x.to(device), iter(testloader).next())
    
        test_prediction = model(flatten_trailing(test_X))
        test_losses.append(
            loss_fn(test_prediction, test_y).item()
        )
    
        test_accuracy.append(
            (test_prediction.argmax(axis=1)==test_y).to(float).mean()
        )
    
        clear_output(wait=True)
    
        print('Accuracy: ', max(test_accuracy))
        plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='train')
        plt.plot(
            np.linspace(0, len(train_losses), len(test_losses) + 1)[1:],
            test_losses, label='test'
        )
        plt.ylabel("Loss")
        plt.xlabel("# steps")
        plt.legend();

        plt.subplot(1, 2, 2)
        plt.plot(test_accuracy, "o-")
        plt.ylabel("Test accuracy")
        plt.xlabel("# epochs");
        plt.show()
    return train_losses, test_losses, test_accuracy


# In[20]:


train_loop(model, num_epochs=10, device=device)


# In[25]:


torch.cuda.is_available()


# In[26]:


torch.cuda.get_device_name()


# In[27]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 1500)
        self.fc2 = nn.Linear(1500, 100)
        self.fc3 = nn.Linear(100, 10)
        
    def forward(self, X):
        Xf = X.flatten(1)
        X1 = F.relu(self.fc1(Xf))
        X2 = F.relu(self.fc2(X1))
        return self.fc3(X2)


# In[28]:


model2 = Net().to(device)


# In[25]:


train_loop(model2, num_epochs=10)


# ## Tensorboard 

# In[29]:


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() # default dir: ./runs/MNIST_Y
# a directory we use to store the intermediate information . 
# everything is sent there.


# In[30]:


train_X, train_y = map(lambda x: x.to(device), iter(trainloader).next())


# In[31]:


img = torch.transpose(
        torch.cat(
            [train_X[train_y==c][:30] for c in range(10)], axis=0
        ).reshape(30, 30, 32, 32),
    1, 2
).reshape(960, 960)


# In[32]:


plt.imshow(img)
plt.show()
print(img.shape)


# Let's send these information to **tensorboard**

# In[33]:


writer.add_image('MNIST', img.view(1, 960, 960))


# In[34]:


writer.close()


# In[35]:


# ! pip install tensorboard


# Now we have to run tensorboard in a separate terminal using the script: `$ tesnorboard --logdir=runs`

# In[36]:


get_ipython().system(' tensorboard --logdir=runs ')
# it will show you binary B & W images 


# In[ ]:




