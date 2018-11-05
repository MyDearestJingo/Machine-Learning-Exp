from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#超参
Learning_Rate=0.001
epoch=10
C=1
batch_size=100

#load train data
x,y=load_svmlight_file("a9a.txt")
x=x.todense()
y = y.reshape((-1, 1))

#load test data
Test_x,Test_y = load_svmlight_file("a9a_t.txt")
Test_x = Test_x.todense()
Test_y = Test_y.reshape((-1, 1))
#加偏执项和初始化w
x=np.column_stack((x,np.ones((x.shape[0],1))))
Test_x=np.column_stack((Test_x,np.zeros((Test_x.shape[0],1))))
Test_x=np.column_stack((Test_x,np.ones((Test_x.shape[0],1))))
w = np.random.random(size=(x.shape[1],1))

#LR:learning rate
import sys
def train(x,y,x_test,y_test,w,C,epoch,batch_size,LR):
    loss_test=[]
    loss_train=[]
    step = x.shape[0] / batch_size
    for e in range(epoch):
        for s in range(int(step)):
            print("step: %3d / %3d" %(e,epoch),end='')
            sys.stdout.write('\r')

            train_X=x[s:s+batch_size]
            train_Y=y[s:s+batch_size]
            w=SVM(train_X,train_Y,w,LR,C)
            loss_train.append(Loss(train_X,train_Y,w,C))
            loss_test.append(Loss(x_test,y_test,w,C))
    return loss_train,loss_test,w

def H_Loss(x,y,w):
    #Hinge loss = ξi = max(0,1−yi(w^T xi + b))
    # print(y.shape)
    # print(np.dot(x,w).shape)
    # print(re.shape)
    return np.maximum(0,1-y*np.dot(x,w))

def Loss(x,y,w,C):
    #The optimization problem becomes:
    # print(np.power(w,2).shape)
    # print(H_Loss(x,y,w).shape)
    loss=1/2*np.power(w,2)+C*np.average(H_Loss(x,y.T,w))
    return np.sum(loss)

def SVM(x,y,w,LR,C):
    y=y.T
    xw=np.dot(x,w)
    xw=(1-y*(xw)>=0)
    gw_batch = -(xw*y*x)
    g_w = np.sum(gw_batch, axis=0)
    g_w =g_w.reshape((-1, 1))
    G = C*g_w+w
    w -= LR*G
    return w

TrainLOSS,TestLOSS,w=train(x,y,Test_x,Test_y,w,C,epoch,batch_size,Learning_Rate)

#draw graph
print(classification_report(Test_y, np.where(np.dot(Test_x, w) > 0, 1, -1),target_names=["positive", "negative"], digits=4))
plt.figure(figsize=(10, 5))
plt.plot(TrainLOSS,label='Loss of Train')
plt.plot(TestLOSS,label='Loss of Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('The graph of Lossing')
plt.legend()
plt.show()
