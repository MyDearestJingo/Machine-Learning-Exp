
# coding: utf-8

# ### Load dataset files

# In[1]:


from sklearn.datasets import load_svmlight_file
X_train, Y_train = load_svmlight_file("a9a.txt", n_features = 123)
X_test, Y_test = load_svmlight_file("a9a_t.txt", n_features = 123)


# ### Preprocess datasets

# In[2]:


import numpy as np 
X_train = X_train.toarray()
X_test = X_test.toarray()
Y_train = Y_train.reshape((-1,1))
Y_test = Y_test.reshape((-1,1))
X_train = np.column_stack((X_train, np.ones((X_train.shape[0],1))))
X_test = np.column_stack((X_test, np.ones((X_test.shape[0],1))))


# ### Define loss function and optimize function

# In[3]:


# loss function
# lambda x,y,w : np.maximum(0,1-y*np.dot(x,w))
def hinge(x,y,w):
    loss = np.maximum(0, 1-y*np.dot(x,w))
    return loss

def loss(x,y,w,C):
    # loss = np.sum(np.power(w,2)/2) + C*np.sum(hinge(x,y,w))
    loss = np.sum(np.power(w,2)/2) + C*np.average(hinge(x,y,w))
    return loss
# optimize function
def grad_descent_svm(x,y,w,learning_rate, C):
    batch_gw = -y*x*(1-y*(np.dot(x,w))>=0)
    gw = np.sum(batch_gw, axis=0)
    g = w + C*gw.reshape((-1,1))
    w -= learning_rate*g
    return w


# In[15]:


import sys
import random

# hyper-para
msgd_learning_rate = 0.0001
msgd_C = 0.6
msgd_max_epoches = 1000
msgd_batch_size = 100
msgd_max_steps = X_train.shape[0]//msgd_batch_size

# MSGD train function
def svm_train_msgd(x_train,y_train,x_test, y_test, C, learning_rate, max_epoches, batch_size, max_steps):
    losses_train = []
    losses_test = []
    
    w = np.ones((x_train.shape[1],1))
    
    for epoch in range(max_epoches):
        for step in range(max_steps):
            temp = step
            step = random.randint(0,x_train.shape[0]-batch_size)
            x = x_train[step:step+batch_size]
            y = y_train[step:step+batch_size]
            w = grad_descent_svm(x,y,w,learning_rate,C)
            step = temp
        losses_train.append(loss(x_train, y_train, w, C))
        losses_test.append(loss(x_test, y_test, w, C))
        print("Epoch: %3d / %3d" %(epoch+1, max_epoches),end='')
        sys.stdout.write('\r')
    print('\n')
    return losses_train, losses_test, w

# hyper-para
gd_learning_rate = 0.0001
gd_C = 0.6
gd_max_epoches = 1000

# FullGD train function
def svm_train_gd(x_train,y_train,x_test, y_test, C, learning_rate, max_epoches):
    losses_train = []
    losses_test = []
    
    w = np.ones((x_train.shape[1],1))
    
    for epoch in range(max_epoches):
        w = grad_descent_svm(x_train,y_train,w,learning_rate,C)
        losses_train.append(loss(x_train, y_train, w, C))
        losses_test.append(loss(x_test, y_test, w, C))
        print("Epoch: %3d / %3d" %(epoch+1, max_epoches),end='')
        sys.stdout.write('\r')
    print('\n')
    return losses_train, losses_test, w


# In[16]:


msgd_losses_train, msgd_losses_test, msgd_w = svm_train_msgd(
    X_train, Y_train, X_test, Y_test,msgd_C, msgd_learning_rate, msgd_max_epoches, msgd_batch_size, msgd_max_steps)
gd_losses_train, gd_losses_test, gd_w = svm_train_gd(
    X_train, Y_train, X_test, Y_test,gd_C,gd_learning_rate, gd_max_epoches)


# In[17]:


from sklearn.metrics import classification_report
print("MSGD Result: ")
print(classification_report(Y_test, np.where(np.dot(X_test, msgd_w) > 0, 1, -1),
                            target_names=["positive", "negative"], digits=4))
print("FullGD Result: ")
print(classification_report(Y_test, np.where(np.dot(X_test, gd_w) > 0, 1, -1),
                            target_names=["positive", "negative"], digits=4))


# In[18]:


import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))
plt.plot(msgd_losses_train, color="r", label="MSGD train")
plt.plot(msgd_losses_test, color="b", label="MSGD test")
plt.plot(gd_losses_train, color="y", label="FullGD train")
plt.plot(gd_losses_test, color="g", label="FullGD test")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("The graph of loss value varing with the number of iterations")
plt.show()

