#%%
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# %%

x, y = torch.load('/home/siuoly/Downloads/MNIST/processed/training.pt')

# %%
plt.imshow(x[3])# didn't put .numpy() into the code 
plt.title(f'Number is {y[3]}')
plt.colorbar()
plt.show()
# %%
y_new = F.one_hot(y , num_classes = 10)
# %%
x.view( -1 , 28**2)

# %%
class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x , self.y = torch.load(filepath)
        self.x = self.x/255.
        # self.y = self.y.to(float)
        self.y = F.one_hot(self.y , num_classes = 10).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index] , self.y[index]
# %%
training_ds = CTDataset('/home/siuoly/Downloads/MNIST/processed/training.pt')
test_ds = CTDataset('/home/siuoly/Downloads/MNIST/processed/test.pt')


# %%
training_dl = DataLoader(training_ds, batch_size=5)

for x,y in training_dl:
    print(x.shape)
    print(y.shape)
    break
len(training_dl)
# %%
L = nn.CrossEntropyLoss()

#%%
# neural network
class MyneuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

f = MyneuralNet()

# %%
# training 
def train_model(dl, f , n_epochs = 20):
    # optimizer
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Training model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x,y) in enumerate(dl):
            # update the weights of the networkopt.zero_grad()
            opt.zero_grad()
            loss_value = L(f(x),y)
            loss_value.backward()
            opt.step()
            # store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)

epoch_data, loss_data = train_model(training_dl,f)


#%%
epoch_avg = epoch_data.reshape(20,-1).mean(axis=1)
loss_avg = loss_data.reshape(20 , -1).mean(axis=1)

#%%
plt.plot(epoch_avg , loss_avg, 'o--')
plt.xlabel('epoch number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (per batch)')
# %% trying
# y_sample = training_ds[0][1]
# yhat_sample = f(training_ds[0][0])
# torch.argmax(yhat_sample)
# %%
xs , ys = training_ds[0:2000]
yhats = f(xs).argmax(axis = 1)

fig, ax = plt.subplots(10 , 4, figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'predict:{yhats[i]}')
fig.tight_layout()
plt.show()
# %% testing
xs , ys = test_ds[0:2000]
yhat = f(xs).argmax(axis = 1)

fig, ax = plt.subplots(10 , 4, figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'predict:{yhat[i]}')
fig.tight_layout()
plt.show()