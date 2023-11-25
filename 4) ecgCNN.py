import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
from sklearn.model_selection import train_test_split
import librosa.display
import pywt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import spectrogram
from scipy.signal import cwt, ricker, morlet, windows
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign


def cwt_spectrum(input_data, fs):

    wavelet = 'morl'
    scales = np.arange(1, 17)

    # Вычисляем CWT
    coefficients, frequencies = pywt.cwt(input_data, scales, wavelet, sampling_period=1 / fs)

    # # Построение графика CWT
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(input_data)
    # plt.subplot(2, 1, 2)
    # plt.imshow((np.abs(coefficients)), aspect='auto', extent=[0, 1, 1, len(scales)], cmap='jet', interpolation='bilinear')
    # plt.colorbar(label='Амплітуда')
    # plt.ylabel('Маштаб')
    # plt.xlabel('Час (секунди)')
    # plt.title('Continuous Wavelet Transform (CWT)')
    # plt.show()
    return coefficients



fs = 500
X_mass = np.load('./Task_X_mass.npy')

Y_mass = np.load('./Task_Y_mass.npy')

X_train, X_test, y_train, y_test = train_test_split(X_mass, Y_mass, train_size=0.7, shuffle=True)

trainloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=50)
testloader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=50)


#
# input_size = np.shape(trainloader.dataset[0][0])[0]*np.shape(trainloader.dataset[0][0])[1]
input_size = np.shape(trainloader.dataset[0][0])[0]
hidden_layers = [128,64]
output_size = 1
epochs = 50

model = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # nn.Conv2d(16, 8, kernel_size=(3, 3), stride=1, padding=1),
    # nn.ReLU(),
    # nn.Conv2d(8, 1, kernel_size=(3, 3), stride=1, padding=1),
    # nn.ReLU(),
    # nn.Flatten(),
    nn.Linear(1500, 500),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, output_size),
    nn.Sigmoid()
)
print(model)
criterion = nn.BCELoss() #Бінарна крос ентропія (перехресна втрата ентропії)
optimizer = optim.SGD(model.parameters(), lr=0.005)

for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:
        images = images.float().unsqueeze(1)
        labels = labels.float()
        optimizer.zero_grad()
        output = model(images).squeeze()
        loss = criterion(output, labels)
        # backward propagation
        loss.backward()
        # update the gradient to new gradients
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Training loss: ", (running_loss / len(trainloader)))




true_count = 0
false_count = 0
for images, labels in testloader:
    labels = labels
    images = images.float().unsqueeze(1)
    output = model(images).squeeze()
    for out_val, lable_val in zip(output, labels):
        if int(round(float(out_val))) == int(lable_val):
            true_count +=1
        else:
            false_count+=1

print(true_count)
print(false_count)
acc = true_count/(true_count+false_count)
print('Accuracy: '+ str(acc))