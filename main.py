import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from CNNmodel import CNN
import os

cat_train = 'D:\\DL-codes\\practice\\computer vision\\cat and dog\\train1\\cat'
dog_train = 'D:\\DL-codes\\practice\\computer vision\\cat and dog\\train1\\dog'

### read the dataset  ###

cat = []
for filename in os.listdir(cat_train):
    img = Image.open(os.path.join(cat_train, filename))
    img_resized = img.resize((512, 512))
    img = np.array(img_resized)
    #print(img.shape)
    cat.append(img)

dog = []
for filename in os.listdir(dog_train):
    img = Image.open(os.path.join(dog_train, filename))
    img_resized = img.resize((512, 512))
    img = np.array(img_resized)
    #print(img.shape)
    dog.append(img)

### img: (512, 512, 3) convert cat and dog to tensor ###
#print(len(cat), len(dog))
cat_tensor = []
for ele in range(len(cat)):
    tmp = cat[ele]
    r = torch.from_numpy(tmp[:,:,0])
    g = torch.from_numpy(tmp[:,:,1])
    b = torch.from_numpy(tmp[:,:,2])
    RGB_tmp = torch.stack((r,g,b), dim = 0)
    #print(RGB_tmp.shape)
    cat_tensor.append(RGB_tmp)

cat_train_data = torch.stack(cat_tensor, dim = 0)
cat_train_data = cat_train_data.float()
#print(cat_train_data.shape)
dog_tensor = []
for ele in range(len(dog)):
    tmp = dog[ele]
    r = torch.from_numpy(tmp[:,:,0])
    g = torch.from_numpy(tmp[:,:,1])
    b = torch.from_numpy(tmp[:,:,2])
    RGB_tmp = torch.stack((r,g,b), dim = 0)
    #print(RGB_tmp.shape)
    dog_tensor.append(RGB_tmp)

dog_train_data = torch.stack(dog_tensor, dim = 0)
dog_train_data = dog_train_data.float()
#print(dog_train_data.shape)

dataset = torch.cat([cat_train_data, dog_train_data], dim = 0)
print("dataset: ", dataset.shape)
label_cat = torch.tensor([1, 0])
label_cat = label_cat.repeat(len(cat), 1)
label_dog = torch.tensor([0, 1])
label_dog = label_dog.repeat(len(dog), 1)
label = torch.cat([label_cat, label_dog], dim = 0)
label = label.float()
print("label: ",label.shape)

### start training ###

model = CNN()
#print(model(dataset).shape)
#print(label.shape)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.00001)

EPOCHS = 500
loss_contain = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(dataset)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    if epoch%20 == 0:
        print(f"epoch: {epoch}: loss: {loss.item()}")
        loss_contain.append(loss.item())

#print(loss_contain)
PATH = 'CNN_CD.pth'
torch.save(model, PATH)
