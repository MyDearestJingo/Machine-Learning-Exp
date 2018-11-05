import pickle
import numpy as np
import sys

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.b_classifier = weak_classifier
        self.n_classifiers = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        # Init the weights of samples to normalization 1
        sample_w_list = np.zeros(X.shape[0], dtype=np.float64)
        sample_w_list[:] = 1.0 / X.shape[0]
        
        self.b_classifiers_list = []
        self.b_classifiers_weights_list = np.zeros(self.n_classifiers, dtype=np.float64)
        self.errors_list = np.ones(self.n_classifiers, dtype=np.float64)

        for m in range(self.n_classifiers):

            print("Training Base Classifier: %4d / %4d" %(m, self.n_classifiers),end='')
            sys.stdout.write('\r')
            
            classifier = self.b_classifier(max_depth=4)
            classifier.fit(X,y,sample_weight=sample_w_list)
            y_pre = classifier.predict(X).reshape(-1,1)
            # error = np.average(sample_w_list*(y_pre!=y),axis=1)
            y.reshape(-1,1)
            print(str("sample_w_list.shape: {0}").format(sample_w_list.shape))
            print(str("y_pre.shape: {0}").format(y_pre.shape))
            print(str("y.shape: {0}").format(y.shape))
            error = np.average(
                y != y_pre, weights=sample_w_list, axis=0)
            print(str("error.shape: {0}").format(error.shape))

            if(error>0.5):
                self.b_classifiers_list.append(classifier)
                self.errors_list[m]=error
                self.b_classifiers_weights_list[m] = 0
                break
            elif(error<=0):
                self.b_classifiers_list.append(classifier)
                self.errors_list[m]=error
                self.b_classifiers_weights_list[m]=0
                break
            
            classifier_weight = 0.5*np.log((1.0-error)/error)
            self.b_classifiers_list.append(classifier)
            self.errors_list[m]=error
            self.b_classifiers_weights_list[m]=classifier_weight
            # _y = y.reshape(-1)*y_pre.reshape(-1)
            # print(_y.shape)
            # print(classifier_weight)
            # e = np.exp(-classifier_weight * _y).reshape(-1)
            # print(e.shape)
            # print(sample_w_list.shape)
            sample_w_list *= np.exp(-classifier_weight * y.reshape(-1) * y_pre.reshape(-1)).reshape(-1)
            sample_w_list /= np.sum(sample_w_list)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_pre = []
        for classifier_weight,classifier in zip(self.b_classifiers_weights_list,self.b_classifiers_list):
            y_pre = classifier.predict(X)
            y_pre.append(classifier_weight*y_pre)
        y_pre = np.sum(np.asarray(y_pre),axis=0) # can be romoved
        return y_pre.reshape(-1,1)

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_pre = []
        for classifier_weight,classifier in zip(self.b_classifiers_weights_list ,self.b_classifiers_list):
            y = classifier.predict(X)
            y_pre.append(classifier_weight*y)
        y_pre = np.sum(np.asarray(y_pre),axis=0) # can be removed
        return np.where(y_pre > 0, 1, -1).reshape(-1,1)

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
