# -*- coding: utf-8 -*-
"""ML-Semester-Lab1-GD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ijO5XOzbOckKpigwMpZB5WVy7Bb3_2DX
"""

from sklearn.datasets import load_svmlight_file as lsf
from sklearn.model_selection import train_test_split as tts
from io import BytesIO
import requests as req
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Best Scheme
val_size = 0.25
learning_rate = 0.00105
max_epoch = 400

learning_rate = 0.01
max_epoch = 50
# regularizer = 0.1

# r = req.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing")

# r = req.get("C:/Users/MyDearest Surface/OneDrive/Documents/ProgramSource/ML_LAB/lab1/housing_scale.txt")
# f = BytesIO(r.content)
x, y = lsf("housing_scale.txt", n_features = 13)

x = x.toarray()
n_samples, n_features = x.shape
x = np.column_stack((np.ones((n_samples,1)),x))
y = y.reshape((-1,1))

x_train, x_val, y_train, y_val = tts(x ,y, test_size=val_size)

def train_GD_with_validation(x_train,y_train,x_val,y_val,max_epoch,learning_rate):
  l_train = []
  l_val = []

  n_samples, n_features = x_train.shape
  w = np.ones((n_features,1))
  w = np.matrix(w)

  for epoch in range(max_epoch):
    G = np.dot(x_train.T, (np.dot(x_train, w)-y_train))
    w += learning_rate*(-G)
    #print(w)
    #
    
    loss_t = np.sum(np.power(np.dot(x_train,w)-y_train,2))/2
    l_train.append(loss_t)

    loss_v = np.sum(np.power(np.dot(x_train,w)-y_train,2))/2
    l_val.append(loss_v)

    #if((epoch+1)%10==0):
      #print(str('Epoch-{0} | Train Loss: {1} | Validation Loss: {2}').format(epoch+1, loss_t, loss_v))
  #if(l_train[-1]>10000): l_train[-1]=0
  #if(l_val[-1]>10000): l_val[-1]=0    
  return l_train[-1], l_val[-1]

#losses_train, losses_val = train_GD_with_validation(x_train, y_train, x_val, y_val, max_epoch, learning_rate)

loss_list_train = []
loss_list_val = []
learningrate_list = []
max_epoch_list = []
for learning_rate in [0.002,0.001,0.0005,0.0001]:
  for max_epoch in [50,100,150,200,250,300]:
    loss_train, loss_val = train_GD_with_validation(x_train,y_train,x_val,y_val,max_epoch,learning_rate)
    print("learning_rate: %.4f | max_epoch: %3d | t_l: %.4f | v_l: %.4f" %(learning_rate, max_epoch, loss_train, loss_val))
    loss_list_train.append(loss_train)
    loss_list_val.append(loss_val)
    learningrate_list.append(learning_rate)
    max_epoch_list.append(max_epoch)
# print(losses)

# csvfile = open("result.csv","wb")
# writer = csv.writer(csvfile)
# # writer.writerow(["Train Loss","Test Loss"])
# writer.writerows(losses)
# csvfile.close()

df = pd.DataFrame({"learning rate": learningrate_list, "max_epoch": max_epoch_list, "train loss":loss_list_train, "validation loss": loss_list_val})
df.to_csv("result.csv", index=False, sep=',')

plt.figure(figsize = (16, 9))
plt.plot(loss_list_train, "-", color = "r", label = 'train loss')
plt.plot(loss_list_val, "-", color = "b", label = 'validation loss')
plt.xlabel("max epochs")
plt.ylabel("loss")
#plt.xticks(np.arange(50,300,50))
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations.")
plt.show()
# plt.savefig("result.png")

