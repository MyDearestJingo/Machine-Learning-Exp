{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_svmlight_file\n",
    "X_train, Y_train = load_svmlight_file(\"a9a.txt\", n_features = 123)\n",
    "X_test, Y_test = load_svmlight_file(\"a9a_t.txt\", n_features = 123)\n",
    "X_train = X_train.toarray()\n",
    "X_test = X_test.toarray()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "X_train = np.column_stack((X_train, np.ones((X_train.shape[0],1))))\n",
    "Y_train = Y_train.reshape((-1,1))\n",
    "X_test = np.column_stack((X_test, np.ones((X_test.shape[0],1))))\n",
    "Y_test = Y_test.reshape((-1,1))\n",
    "w = np.ones((X_train.shape[1],1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# hyper-para\n",
    "learning_rate = 0.0001\n",
    "C = 0.5\n",
    "max_epoches = 1\n",
    "batch_size = 10\n",
    "max_steps = X_train.shape[0]//batch_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# loss function\n",
    "def hingeloss(x,y,w):\n",
    "    loss = np.maximum(0, 1-y*np.dot(x,w))\n",
    "    return loss\n",
    "\n",
    "def loss(x,y,w):\n",
    "    loss = np.sum(np.power(w,2)/2) + C*np.sum(hingeloss(x,y,w))\n",
    "    return loss\n",
    "\n",
    "losses_train = []\n",
    "losses_test = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for epoch in range(max_epoches):\n",
    "    for step in range(max_steps):\n",
    "        x = X_train[step:step+batch_size]\n",
    "        y = Y_train[step:step+batch_size]\n",
    "        \n",
    "        # print(y.shape)\n",
    "        # print(np.dot(x,w).shape)\n",
    "        batch_gw = -y*x*(1-y*np.dot(x,w)>=0) \n",
    "        gw = np.sum(batch_gw,axis=0)\n",
    "        w = w+C*gw\n",
    "        G = -w\n",
    "        w += learning_rate*G\n",
    "\n",
    "        losses_train.append(loss(x,y,w))\n",
    "        losses_test.append(loss(X_test,Y_test,w))\n",
    "    print(str(\"Epoch: %3d | Train Loss:    %.2f | Test Loss:   %.2f\" %(epoch, losses_train[epoch*batch_size], losses_train[epoch*batch_size])))\n",
    "    \n",
    "print(\"complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
