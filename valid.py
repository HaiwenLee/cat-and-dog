import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import os
from CNNmodel import CNN


PATH = 'D:\\DL-codes\\practice\\computer vision\\cat and dog\\CNN_CD.pth'
model = torch.load(PATH)
model.eval()

cat_test = 'D:\\DL-codes\\practice\\computer vision\\cat and dog\\test1\\cat'
dog_test = 'D:\\DL-codes\\practice\\computer vision\\cat and dog\\test1\\dog'

cat = []
for filename in os.listdir(cat_test):
    img = Image.open(os.path.join(cat_test, filename))
    img_resized = img.resize((512, 512))
    img = np.array(img_resized)
    cat.append(img)

dog = []
for filename in os.listdir(dog_test):
    img = Image.open(os.path.join(dog_test, filename))
    img_resized = img.resize((512, 512))
    img = np.array(img_resized)
    dog.append(img)
cat_tensor = []
for ele in range(len(cat)):
    tmp = cat[ele]
    r = torch.from_numpy(tmp[:,:,0])
    g = torch.from_numpy(tmp[:,:,1])
    b = torch.from_numpy(tmp[:,:,2])
    RGB_tmp = torch.stack((r,g,b), dim = 0)
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
    dog_tensor.append(RGB_tmp)

dog_train_data = torch.stack(dog_tensor, dim = 0)
dog_train_data = dog_train_data.float()
#print(dog_train_data.shape)

test_dataset = torch.cat([cat_train_data, dog_train_data], dim = 0)
#print(model(test_dataset))
#print(test_dataset.shape[0])
cata = []
for i in range(test_dataset.shape[0]):
    key = model(test_dataset)[i]
    if key[0]>key[1]:
        cata.append('cat')
    else:
        cata.append('dog')

print("test result: ", cata)


