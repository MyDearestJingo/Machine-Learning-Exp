from feature import NPDFeature
import numpy as np
import cv2
import os
import pickle
import sys

pos_img = "datasets/original/face"
neg_img = "datasets/original/nonface"

n_face = 0
n_noface = 0
for file in os.listdir(pos_img):
    n_face+=1
for file in os.listdir(neg_img):
    n_noface+=1 

pos_ds = []
n=0
for file in os.listdir(pos_img):
    n+=1
    print("Face: %4d / %4d" %(n, n_face),end='')
    sys.stdout.write('\r')
    img = cv2.imread(os.path.join(pos_img,file),cv2.IMREAD_GRAYSCALE) # load image file in Gray_mode
    img = cv2.resize(img,(24,24))
    feature = NPDFeature(img).extract()
    pos_ds.append(feature)
pos_ds = np.asarray(pos_ds)
print(pos_ds.shape)

neg_ds = []
n=0
for file in os.listdir(neg_img):
    n += 1
    print("No Face: %4d / %4d" %(n, n_noface),end='')
    sys.stdout.write('\r')
    img = cv2.imread(os.path.join(neg_img,file),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(24,24))
    feature = NPDFeature(img).extract()
    neg_ds.append(feature)
neg_ds = np.asarray(neg_ds)

X = np.concatenate((pos_ds,neg_ds)) # concatenate positive and negtive dataset
y_pos = np.ones((pos_ds.shape[0],1))
y_neg = -np.ones((neg_ds.shape[0],1))
y = np.concatenate((y_pos,y_neg))
dataset = np.concatenate((X,y), axis = 1)

with open("dataset.pkl","wb") as file: pickle.dump(dataset,file)
