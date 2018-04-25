
# coding: utf-8

# # ML Project2

# In[122]:

'''

loading input - output data and explore the dataset

'''
from scipy.io import loadmat
from sklearn import decomposition

import matplotlib.pyplot as plt

inputDataPath  = 'data/Proj2FeatVecsSet1.mat'
outputDataPath = 'data/Proj2TargetOutputsSet1.mat'

inputDataObj  = loadmat(inputDataPath)
outputDataObj = loadmat(outputDataPath)

inputData  = inputDataObj['Proj2FeatVecsSet1']
outputData = outputDataObj['Proj2TargetOutputsSet1']

pca = decomposition.PCA(n_components  = 30)
X = pca.fit_transform(inputData)

print X[0]

data = zip(X, outputData)



# In[123]:

"""

computes confusion matrix

@param   Y                   predicted labels

@param   ClassLabels         actual / true labels  

""" 

from sklearn.metrics import confusion_matrix

def MyConfusionMatrix(Y, ClassNames):
    return confusion_matrix(Y, ClassNames)    


# In[124]:

"""

training script

"""

import numpy as np
import time

from sklearn.svm import SVC
from skrvm import RVC

from sklearn.model_selection import GridSearchCV

def MyTrainClassifier(XEstimate, XValidate, Parameters):
    
    X_train, Y_train = zip(*XEstimate)
        
    X_train = np.array(list(X_train))
    Y_train = np.array([np.where(output == 1)[0][0] for output in list(Y_train)])
    
    # sampling a small amount of training data for finding optimal hyper-parameters
    X_hyper = X_train[:500, :]
    Y_hyper = Y_train[:500]
    
    X_validate, Y_validate = zip(*XValidate)
        
    X_validate = np.array(list(X_validate))
    Y_validate = np.array([np.where(output == 1)[0][0] for output in list(Y_validate)])
    
    # SVM
    if Parameters['algorithm'] == 'SVM':
        
        SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, False)
    
    elif Parameters['algorithm'] == 'RVM':
        
        RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate)        
    
    elif Parameters['algorithm'] == 'GPR':
     
        Gaussian(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, False)


# In[125]:

def SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train):
    
    hyper_param_grid = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]

    estimator = GridSearchCV(SVC(decision_function_shape='ovo'), hyper_param_grid, cv=3, scoring='precision_macro')    
    estimator.fit(X_hyper, Y_hyper) 
    
    clf = estimator.best_estimator_
    print "found best estimator, training the estimator"
    
    if train:
        clf.fit(X_train, Y_train)
        writeObj('svm_model.pkl', clf)
    else:
        clf = readObj('svm_model.pkl')
        print clf.score(X_validate, Y_validate)


# In[126]:

def RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate):
    clf = RVC(n_iter=1, kernel='linear')

    start = time.clock()

    clf.fit(X_train, Y_train)
    
    writeObj('rvm_model.pkl', clf)
    
    print time.clock() - start, "s"
    print clf.score(X_validate, Y_validate)


# In[127]:

from sklearn.multiclass import OneVsOneClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsRestClassifier
import pickle

def Gaussian(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train):
    print "gaussian in progress"
        
    if train:
        kernel_rbf = 1.0 * RBF()
    
        clf = GaussianProcessClassifier(kernel=kernel_rbf, multi_class='one_vs_rest')
        clf.fit(X_train[:1000, :], Y_hyper[:1000])

        writeObj('gaussian_model', clf)
    else:
        clf = readObj('gaussian_model')
        
        print clf.score(X_validate[:500, :], Y_validate[:500])
    #print clf.predict_proba([X_validate[0]])
    
    #myclf = OVO('gaussian')
    #myclf.fit(X_train, Y_train)
    
    #print myclf.predict(X_validate[:500, :])
    #print Y_validate[:500]


# In[118]:

def writeObj(name, obj):
    with open(name, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[119]:

def readObj(name):
    with open(name, 'rb') as input:
        clf = pickle.load(input)
    
    return clf


# In[120]:

"""

 K-fold cross validation script

"""
from sklearn.model_selection import KFold
from random import shuffle

def MyCrossValidate(XTrain, Nf):
    shuffle(XTrain)
    kf = KFold(n_splits = Nf)
    
    j = 1
    
    for train_index, test_index in kf.split(XTrain):
        En = [XTrain[i] for i in train_index]
        Vn = [XTrain[i] for i in test_index]
        
        print "fold {} in progress".format(j)
        
        MyTrainClassifier(En, Vn, {'algorithm':'RVM'})
        
        j = j + 1


# In[121]:

MyCrossValidate(data, 5)


# In[ ]:

class OVO:
    def __init__(self, model):
        self.model_ = model
        
    def fact(self, n):
        if n == 0:
            return 1
        
        return n*self.fact(n-1)

    def nCr(self, n, r):
        return self.fact(n)/(self.fact(n-r)*self.fact(r))
    
    def getModel(self):
        if self.model_ == 'gaussian':
            return GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        
    def fit(self, X, Y):
        self.Nclasses_     = len(np.unique(Y))
        self.Nclassifiers_ = self.nCr(self.Nclasses_, 2)
        
        Nclasses = self.Nclasses_
        Nclassifiers = self.Nclassifiers_
        
        dataparts = [None]*Nclasses
        classifiers = [[None]*Nclasses]*Nclasses
        
        print classifiers
        
        for i in range(Nclasses):
            dataparts[i] = np.where(Y == i)[0]
            
        for i in range(Nclasses):
            for j in range(i+1, Nclasses):
                print "training classifier: ", i, " ",j
                
                xi = X[dataparts[i]]
                xj = X[dataparts[j]]
                
                yi = [0]*len(xi)
                yj = [1]*len(xj)
                
                x = np.vstack([xi, xj])
                y = np.hstack([yi, yj])
                
                clf = self.getModel() 
                
                print "clf fitting"
                clf.fit(x, y)
                print "clf fitting done"
                
                classifiers[i][j] = clf
        
        self.classifiers = classifiers
        #print classifiers
    
    def predict(self, X):
        Nclasses = self.Nclasses_
        Nclassifiers = self.Nclassifiers_
        
        classifiers = self.classifiers
        
        Y = []
        
        for x in X:
            probabilities = [0]*Nclasses

            for i in range(Nclasses):
                for j in range(i+1, Nclasses):
                    clf = classifiers[i][j]
                
                    probabilities[i] += clf.predict_proba(X)[0][0]
                    probabilities[j] += clf.predict_proba(X)[0][1]
                
            Y.append(probabilities.index(max(probabilities)))
        
        return Y


# In[ ]:



