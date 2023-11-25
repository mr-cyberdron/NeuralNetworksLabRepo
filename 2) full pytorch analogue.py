import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time


# Перевірити доступність GPU
if torch.cuda.is_available():
    # Встановити пристрій за замовчуванням на GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    # Якщо GPU недоступний, використовувати CPU
    device = torch.device("cpu")

# Перевести всі тензори та моделі на вибране пристрій
torch.set_default_device("cpu")

input_layer_size = 28*28
learning_rate = 0.01
momentum_factor = 0.9
train_batch_size = 50
test_batch_size = 10
epochs = 250

def load_data(train_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True,)
    # generator=torch.Generator(device='cuda'))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True,)
    # generator=torch.Generator(device='cuda'))

    return train_loader, test_loader



class MnistClassifNet(nn.Module):
    def __init__(self):
        super(MnistClassifNet,self).__init__()
        self.fc1 = nn.Linear(input_layer_size, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,10)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return F.sigmoid(self.fc3(x))

netInstance = MnistClassifNet()
print(netInstance)
optimizer = optim.SGD(netInstance.parameters(),lr=learning_rate,momentum=momentum_factor)
lossFunction = nn.NLLLoss()   #BCELoss()
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

for epoch in range(epochs):
    start = time.time()

    for batch_idx, (data_batch, labels) in enumerate(train_loader):
        data_batch_var =data_batch#.cuda()
        labels_var = labels#.cuda()
        data_batch_var = data_batch_var.view(-1,input_layer_size)
        optimizer.zero_grad()
        forward_pass_result = netInstance(data_batch_var)
        loss_result = lossFunction(forward_pass_result, labels_var)
        loss_result.backward()
        optimizer.step() # make one optimisation step

        if batch_idx%10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_batch_var), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), float(loss_result.data)))
    end = time.time() - start
    print(f'The network training time: {end} sec')





