
# coding: utf-8

# # ML Project2

# In[9]:

# utility methods

def reduce_dimensions(data):
    pca = decomposition.PCA(n_components = 7)
    
    X = pca.fit_transform(data)
    return X

def readObj(name):
    with open(name, 'rb') as input:
        clf = pickle.load(input)
    
    return clf

def writeObj(name, obj):
    with open(name, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[10]:

'''

loading input - output data and explore the dataset

'''
from scipy.io import loadmat
from sklearn import decomposition

inputDataPath  = 'data/Proj2FeatVecsSet1.mat'
outputDataPath = 'data/Proj2TargetOutputsSet1.mat'

inputDataObj  = loadmat(inputDataPath)
outputDataObj = loadmat(outputDataPath)

inputData  = inputDataObj['Proj2FeatVecsSet1']
outputData = outputDataObj['Proj2TargetOutputsSet1']


data = zip(inputData, outputData)


# In[11]:

"""

computes confusion matrix

@param   Y                   predicted labels

@param   ClassLabels         actual / true labels

"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd

def MyConfusionMatrix(Y, ClassNames):
    ClassLabels = list(np.unique(ClassNames))
    
    conf_matrix = confusion_matrix(Y, ClassNames)
    accuracy = accuracy_score(Y, ClassNames)
    
    columns = tuple(ClassLabels)
    rows = tuple(ClassLabels)
    
    df = pd.DataFrame(data=conf_matrix, columns=ClassLabels)
    
    print "\nconfusion matrix: \n"
    print df
    
    return conf_matrix, accuracy


# In[12]:

def SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train):

    hyper_param_grid = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]

    estimator = GridSearchCV(SVC(decision_function_shape='ovr'), hyper_param_grid, cv=3, scoring='precision_macro')
    
    print "SVM: executing grid search to find optimal hyper-parameters"
    
    estimator.fit(X_hyper, Y_hyper)

    clf = estimator.best_estimator_
    
    print "found best hyperparameters:"
    print estimator.best_params_
    print "training the estimator"

    if train:
        clf.fit(X_train, Y_train)
        writeObj('svm_model.pkl', clf)
        
        Y_pred = clf.predict(X_validate)
        return Y_pred, clf
    
    else:
        clf = readObj('svm_model.pkl')
        print clf.score(X_validate, Y_validate)


# In[13]:

def RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train):
    clf = OneVsRestClassifier(RVC(n_iter=1))
    start = time.clock()
    
    X_train_reduced = reduce_dimensions(X_train)
    X_validate_reduced = reduce_dimensions(X_validate)
    
    if train:
        clf.fit(X_train_reduced[:5000, :], Y_train[:5000])
        writeObj('rvm_model.pkl', clf)
    else:
        clf = readObj('rvm_model.pkl')
        print clf.score(X_validate_reduced, Y_validate)

    print time.clock() - start, "s"
    print clf.predict_proba(X_validate[0])


# In[14]:

from sklearn.multiclass import OneVsOneClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsRestClassifier
import pickle

def Gaussian(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train):
    print "GPR:"
    
    X_train_reduced = reduce_dimensions(X_train)
    X_validate_reduced = reduce_dimensions(X_validate)
    
    if train:
        kernel_rbf = 1.0 * RBF()

        clf = GaussianProcessClassifier(kernel=kernel_rbf, multi_class='one_vs_rest')
        clf.fit(X_train_reduced[:1000, :], Y_train[:1000])

        writeObj('gaussian_model', clf)
    else:
        clf = readObj('gaussian_model')

        print clf.score(X_validate_reduced[:500, :], Y_validate[:500])


# In[15]:

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
    
    train = Parameters['training_mode']
    
    # SVM
    if Parameters['algorithm'] == 'SVM':

        Y_predict, model = SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train)

    elif Parameters['algorithm'] == 'RVM':

        Y_predict, model = RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train)

    elif Parameters['algorithm'] == 'GPR':

        Y_predict, model = Gaussian(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, train)

    return Y_predict, {'model' : model}


# In[16]:

"""

 K-fold cross validation script

"""
from sklearn.model_selection import KFold
from random import shuffle

def MyCrossValidate(XTrain, Nf):
    shuffle(XTrain)
    kf = KFold(n_splits = Nf)
    
    j = 1
    
    EstParameters = []
    EstConfMatrices = []
    accuracies = []
    
    for train_index, test_index in kf.split(XTrain):
        En = [XTrain[i] for i in train_index]
        Vn = [XTrain[i] for i in test_index]
        
        print "\nfold {} in progress:\n".format(j)
        
        Y_predicted, EstParameter = MyTrainClassifier(En, Vn, {'algorithm':'GPR', 'training_mode':True})
        
        _, Y_validate = zip(*Vn)
        Y_validate = np.array([np.where(output == 1)[0][0] for output in list(Y_validate)])
        
        Cn, acc = MyConfusionMatrix(Y_predicted, Y_validate)
        
        EstConfMatrices.append(Cn)
        EstParameters.append(EstParameter)
        
        accuracies.append(acc)
        
        j = j + 1
    
    print ""
    
    best_model_idx = accuracies.index(max(accuracies))
    best_model = EstParameters[best_model_idx]['model']
    
    X, Y = zip(*XTrain)
    
    X = np.array(list(X))
    Y = np.array([np.where(output == 1)[0][0] for output in list(Y)])
    
    YTrain = best_model.predict(X)
    
    print "overall confusion matrix :"
    
    ConfMatrix, acc = MyConfusionMatrix(YTrain, Y)
    
    return YTrain, EstParameters, EstConfMatrices, ConfMatrix


# In[17]:

MyCrossValidate(data, 5)


# In[ ]:



