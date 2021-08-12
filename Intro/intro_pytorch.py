# -*- coding: utf-8 -*-
"""Intro-Pytorch

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l_xwW3Sbze-j-EVhOTnGgmNrPlkUKiVr

# **Intro to Pytorch and Tensorboard**
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tqdm
from torch.utils.tensorboard import SummaryWriter

# %matplotlib inline

"""
## Data load

All dataset classes are subclasses of [`torch.util.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) have `__getitem__` and `__len__` methods implemented. Thus, it can be passed to a [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), which can load multiple samples parallelly.
Popular Dataset subclasses:

- [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)
- [Fashion-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)
- [CIFAR](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)
- [CelebA](https://pytorch.org/docs/stable/torchvision/datasets.html#celeba)

`MNIST` constructor signature: `torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)`, where `transform` - function/transform that takes in an PIL (Height x Width x Channels) image and returns a transformed version. 

**Several transformations can be combined together. Popular transformations**:
- `torchvision.transforms.Normalize(mean, std, inplace=False)`
- `torchvision.transforms.ToTensor()` - Converts a PIL Image or `numpy.ndarray` (H x W x C) in the range `[0, 255]` to a `torch.FloatTensor` of shape (C x H x W) in the range `[0.0, 1.0]` 


DataLoader constructor signature
`torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)`, where
- dataset is DataSet instance
- batch_size - number of items sampled at every iteration
- num_workers - number of simultaneous reading processes (**NB** on Windows you might want to set it to `0`)

DataLoaders provide convenient interface for training loops: 

```python
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            ...
            
```
or 
```python
    batch_X, batch_y = iter(trainloader).next()
```"""

from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) # normalize around 0.5 with std 0.5
]) # Transforms data to tensor. ('numpy array' or 'PIL image' --> Tensor)

ds_train = datasets.MNIST("~/share/data/", train=True, download=True, transform=transform)
ds_test = datasets.MNIST("~/share/data/", train=False, download=True, transform=transform)

#dataloader: Allows you wrap an iterable around the Dataset instance, to enable 
# easy access to the sample. here we have 512 batch per an iteration.
# briefly, it take sample form our Dataset, here is MNIST
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=512,
                                           shuffle=True, num_workers=2)
# batch_size = number of objects per iteration

test_loader = torch.utils.data.DataLoader(ds_test, batch_size=1000, 
                                          shuffle=False, num_workers=2)

print('Train:', ds_train, "\nTest:", ds_test)

type(train_loader)

"""**Get sample out of the train loader**  
it gives you couple of data ---→ ***X_batch*** & ***y_batch***
"""

X_batch, y_batch = iter(train_loader).next() # The next() function returns the 
# next item in an iterator. 1 iteration = 512 sample

print('batch size:', len(X_batch), "batch_dimension", X_batch.shape)

"""The output is torch tensor"""

type(X_batch), type(ds_train)

plt.figure(figsize=(4, 4), dpi=100)
plt.axis('off')

plt.imshow(
    torch.transpose(
        torch.cat(
            [X_batch[y_batch == c][:10] for c in range(10)], axis=0
        ).reshape(10, 10, 28, 28),
        1, 2
    ).reshape(280, 280)
)
plt.show()

"""## Automatic differentiation
Automatic differentiaion is the main mechanism for the backpropagation in PyTorch. PyTorch provides a module, `autograd`, for automatically calculating the gradients of tensors. We can use it to calculate the gradients of all our parameters with respect to the loss. Autograd works by keeping track of operations performed on tensors, then going backwards through those operations, calculating gradients along the way. To make sure PyTorch keeps track of operations on a tensor and calculates the gradients, you need to set `requires_grad = True` on a tensor. You can do this at creation with the `requires_grad` keyword, or at any time with `x.requires_grad_(True)`.

You can turn off gradients for a block of code with the `torch.no_grad()` context:

```
x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False
```
Also, you can turn on or off gradients altogether with `torch.set_grad_enabled(True|False)`.

The gradients are computed with respect to some variable `z` with `z.backward()`. This does a backward pass through the operations that created `z`.
"""

x = torch.randn(2, 2, requires_grad=True) # 2*2 matrix, initialized with random number
print(x)

# operation on x  
y = x ** 2
y1 = torch.sqrt(x) * torch.sin(2 * x)
print(y, y1)

"""Below we can see the operation that created `y`, a power operation `PowBackward0`."""

## grad_fn shows the function that generated this variable y
print('Operation on x in y1 is : ', y1.grad_fn)
print('---'*10)
print('Operation on x in y is : ', y.grad_fn)

"""Till now, `autograd` has kept track of the operators on x, which is `<PowBackward0`. Pytorch would not compute the gradient unless you use `backward()` function."""

print(x.grad)

"""Sine we have not computed any gradient of y with respect to x"""

#y.backward()

"""The autograd module keeps track of these operations and knows how to calculate the gradient for each one. In this way, it's able to calculate the gradients for a chain of operations, with respect to any one tensor. In order to get rid of this error, Let's reduce the tensor `y` to a scalar value, the mean.

To calculate the gradients, you need to run the `.backward` method on a variable `z`, for example. This will calculate the gradient for `z` with respect to `x`

$$\frac{\partial z}{\partial x}=\frac{\partial}{\partial x}\left[\frac{1}{n} \sum_{i}^{n} x_{i}^{2}\right]=\frac{x}{2}$$
"""

z = y.mean() # sum(1/4 (x ** 2))
print('z = ', z, '\n','---'*15)
z.backward() #1/4 ( d/dx x ** 2)
print(x.grad)
print('---'*15, '\nx/2 =', x/2)

print(x)

print(y)

0.9129/4 + 0.149/4 + 0.9854/4 + 0.0194/4

"""***

Deep learning requires rapid calculation. Fortunately, Pytorch supports GPU acceleration. First, you have to check if your graphic card supports CUDA. for install and more details see [here](https://developer.nvidia.com/cuda-downloads).  
Below, we define a function to choose a cuda randomly. 
"""

def get_random_gpu():
    return np.random.randint(4)

def get_free_gpu():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
    nvmlInit()

    return np.argmax([
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(nvmlDeviceGetCount())
    ])

if torch.cuda.is_available():
    cuda_id = get_random_gpu()
    if cuda_id == 0:
      device = f'cuda:{cuda_id}'
      print(f'Selected {device}')
    else:
      device = 'cpu'
      print('WARNING: using cpu!')

"""**From now, we send our models and data to selected device** `cuda:0`"""

# create a network that stacks layers on top of each other
model = nn.Sequential(
    nn.Linear(784, 100), # add first "dense" layer with 784 input
                         # units and 100 output units (hidden layer
                         # with 100 neurons).
    nn.ReLU(),
    nn.Linear(100, 10), # "dense" layer with 10 output
                        # units (for 10 classes).
).to(device)

print("Weight shapes:")
for w in model.parameters():
    print("  ", w.shape)

for i in range(3):
  print(model[i])

def flatten_trailing(batch):
    flat = batch.flatten(1)
    #raise NotImplementedError()
    return flat

loss_fn = nn.CrossEntropyLoss()
predictions = model(flatten_trailing(X_batch.to(device)))

loss = loss_fn(predictions, y_batch.to(device))

print('Before backward pass: \n', model[2].weight.grad)

loss.backward()

print('After backward pass: \n', model[2].weight.grad[0])

values = ["a", "b", "c"]
 
for count, value in enumerate(values):
  print(count, value)

def train(model, num_epochs=1, batch_size=512, loss_fn=loss_fn, device='cpu'):
    # some quantities to plot
    train_losses = []
    test_losses = []
    test_accuracy = []
    model_dev = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i_epoch in range(num_epochs):
        t = tqdm.tqdm(iter(train_loader), leave=False, total=len(train_loader))
        for idx, data in enumerate(t):
            # get the next chunk (batch) of data:
            # Sending data to 'device' with 'map'
            batch_X, batch_y = map(lambda x: x.to(device), data)

            # all the black magic:
            loss = loss_fn(model(flatten_trailing(batch_X)), batch_y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            # remember the loss value at this step
            train_losses.append(loss.item())

        # evaluate test loss and metrics
        test_X, test_y = map(lambda x: x.to(device), iter(test_loader).next())

        test_prediction = model(flatten_trailing(test_X.to(device)))
        test_losses.append(
            loss_fn(test_prediction, test_y).item()
        )
        test_accuracy.append(
            (test_prediction.argmax(axis=1) == test_y).to(float).mean()
        )
        # all the rest is simply plotting

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

train(model, device=device, num_epochs=10);

"""### Alternative way to define neuralnetwork configuration is **nn.Module**"""

# Alternate method to define the same model as previous one
class Net(nn.Module):
  def __init__(self): # constructor : defining all objects and methods 
    super(Net, self).__init__() # In order to have access to parants methods and objects
    self.fc1 = nn.Linear(784, 100)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, X):
    Xf = X.flatten(1) # adding flatten here
    X1 = F.relu(self.fc1(Xf))  # first dense layer
    return self.fc2(X1) # second layer without activator

model2 = Net().to(device)

train(model2, num_epochs=10, device=device)

model2.parameters()