
# coding: utf-8

# In[7]:


from sklearn.datasets import load_svmlight_file
X_train, Y_train = load_svmlight_file("a9a.txt", n_features = 123)
X_test, Y_test = load_svmlight_file("a9a_t.txt", n_features = 123)
X_train = X_train.toarray()
X_test = X_test.toarray()


# In[8]:


import numpy as np 
X_train = np.column_stack((X_train, np.ones((X_train.shape[0],1))))
Y_train = Y_train.reshape((-1,1))
X_test = np.column_stack((X_test, np.ones((X_test.shape[0],1))))
Y_test = Y_test.reshape((-1,1))
w = np.ones((X_train.shape[1],1))


# In[9]:


# hyper-para
learning_rate = 0.002
C = 0.7
max_epoches = 40
batch_size = 100
max_steps = X_train.shape[0]//batch_size


# In[10]:


# loss function
def hingeloss(x,y,w):
    loss = np.maximum(0, 1-y*np.dot(x,w))
    return loss

def loss(x,y,w,C):
    #loss = np.sum(np.power(w,2)/2) + C*np.sum(hingeloss(x,y,w))
    loss = np.sum(np.power(w,2)/2) + C*np.average(hingeloss(x,y,w))
    return loss

losses_train = []
losses_test = []


# In[ ]:

import sys
for epoch in range(max_epoches):
    for step in range(max_steps):

        print("step: %3d" %(step),end='')
        sys.stdout.write('\r')

        x = X_train[step:step+batch_size]
        y = Y_train[step:step+batch_size]
        
        # print(y.shape)
        # print(np.dot(x,w).shape)
        batch_gw = -y*x*(1-y*np.dot(x,w)>=0) 
        gw = np.sum(batch_gw,axis=0)
        d = w+C*gw.reshape(-1,1)
        G = -d
        w += learning_rate*G

    losses_train.append(loss(x,y,w,C))
    losses_test.append(loss(X_test,Y_test,w,C))
    #print(str("Epoch: %3d | Train Loss:    %.4f | Test Loss:   %.4f" %(epoch, losses_train[epoch], losses_train[epoch])))
    
print("complete")

from sklearn.metrics import classification_report

print(classification_report(Y_test, np.where(np.dot(X_test, w) > 0, 1, -1),
                            target_names=["positive", "negative"], digits=4))

import matplotlib.pyplot as plt
plt.figure(figsize = (16, 9))
plt.plot(losses_train, "-", color = "r", label = 'train loss')
plt.plot(losses_test, "-", color = "b", label = 'test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations.")
plt.show()
