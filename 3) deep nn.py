import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import optim
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])]) #!!!!
trainset = torchvision.datasets.MNIST('./MNIST_data/', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

input_size = trainloader.dataset.train_data.shape[1] * trainloader.dataset.train_data.shape[2]
hidden_layers = [128,64]
output_size = 10
epochs = 5

model = nn.Sequential(
    nn.Linear(input_size, hidden_layers[0]),
    nn.ReLU(),
    nn.Linear(hidden_layers[0], hidden_layers[1]),
    nn.ReLU(),
    nn.Linear(hidden_layers[1], output_size),
    nn.LogSoftmax(dim=1)
)
print(model)
criterion = nn.NLLLoss() #The negative log likelihood loss
optimizer = optim.SGD(model.parameters(), lr=0.003)

for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:
        # Flatten the Image from 28*28 to 784 column vector
        images = images.view(images.shape[0], -1)

        # setting gradient to zeros
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        # backward propagation
        loss.backward()

        # update the gradient to new gradients
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Training loss: ", (running_loss / len(trainloader)))

def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

# Getting the image to test
images, labels = next(iter(trainloader))
# Flatten the image to pass in the model
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)
# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
view_classify(img, ps)