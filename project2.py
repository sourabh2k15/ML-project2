
# coding: utf-8

# # ML Project2

# In[2]:

"""

Utility methods

"""

# do PCA to reduce dimensionality, required for RVM and GPR
def reduce_dimensions(data):
    pca = decomposition.PCA(n_components = 7)
    
    X = pca.fit_transform(data)
    return X

# reads model from pickled object file
def readObj(name):
    with open('transfer_learning/' + name, 'rb') as input:
        clf = pickle.load(input)
    
    return clf

# writes model to a pickled object file
def writeObj(name, obj):
    with open(name, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def zipData(X, Y):
    return zip(X, Y)

def unzipData(X):
    x, y = zip(*X)

    x = np.array(list(x))
    y = np.array([np.where(output == 1)[0][0] for output in list(y)])
    
    return x,y


# In[3]:

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


# In[4]:

"""

pretty prints confusion matrix and returns confusion matrix and accuracy score

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
    
    print "\n"
    print "accuracy: ", accuracy
    
    return conf_matrix, accuracy


# In[5]:

"""
SVM (Support Vector Machine):

performs grid search to compute optimal hyper-parameters
uses those hyper-parameters for the estimator, fits it on the training data

returns trained model and writes it to file for transfer learning

"""

def SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params):

    hyper_param_grid = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]

    estimator = GridSearchCV(SVC(decision_function_shape='ovr'), hyper_param_grid, cv=3, scoring='precision_macro')
    
    print "SVM: executing grid search to find optimal hyper-parameters"
    
    estimator.fit(X_hyper, Y_hyper)

    clf = estimator.best_estimator_
    train = params['train']
    
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


# In[6]:

"""
RVM (Relevance Vector Machine):

uses PCA to reduce dimensionality as RVM training takes a long time
also uses a subset of training data to save time


returns trained model and writes it to file for transfer learning

"""

def RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params):
    clf = OneVsRestClassifier(RVC(n_iter=1))
    start = time.clock()
    
    X_train_reduced = reduce_dimensions(X_train)
    X_validate_reduced = reduce_dimensions(X_validate)
    
    train_size = params['train_size']
    test_size  = params['test_size']
    train      = params['train']
    
    if train:
        clf.fit(X_train_reduced[:train_size, :], Y_train[:train_size])
        writeObj('rvm_model.pkl', clf)
        
        Y_pred = clf.predict(X_validate_reduced[:test_size])
        return Y_pred, clf
    else:
        clf = readObj('rvm_model.pkl')
        Y_pred = clf.predict(X_validate_reduced[:test_size])
        return Y_pred, clf

    print "training took ", time.clock() - start, "s"


# In[7]:

"""
GPR (Gaussian Process Regressor):

uses PCA to reduce dimensionality as training takes a long time
also uses a subset of training data to save time

returns trained model and writes it to file for transfer learning

"""

from sklearn.multiclass import OneVsOneClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsRestClassifier
import pickle

def GPR(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params):
    print "GPR training :"
    
    X_train_reduced = reduce_dimensions(X_train)
    X_validate_reduced = reduce_dimensions(X_validate)
    
    train_size = params['train_size']
    test_size  = params['test_size']
    train      = params['train']
    
    if train:
        start = time.clock()
        kernel_rbf = 1.0 * RBF()
        
        clf = GaussianProcessClassifier(kernel=kernel_rbf, multi_class='one_vs_rest')
        clf.fit(X_train_reduced[:train_size, :], Y_train[:train_size])

        writeObj('gaussian_model.pkl', clf)
        print "training took ", time.clock() - start, " s"
        
        Y_pred = clf.predict(X_validate_reduced[:test_size])
        return Y_pred, clf
    else:
        clf = readObj('gaussian_model.pkl')
        Y_pred = clf.predict(X_validate_reduced[:test_size])
        
        return Y_pred, clf


# In[8]:

import numpy as np
import time

from sklearn.svm import SVC
from skrvm import RVC

from sklearn.model_selection import GridSearchCV

def MyTrainClassifier(XEstimate, XValidate, Parameters):
    
    X_train, Y_train = unzipData(XEstimate)

    # sampling a small amount of training data for finding optimal hyper-parameters
    X_hyper = X_train[:500, :]
    Y_hyper = Y_train[:500]

    X_validate, Y_validate = unzipData(XValidate)

    train = Parameters['training_mode']
    params = { 'train' : train, 'train_size' : 20, 'test_size' : 20 }
    
    if Parameters['algorithm'] == 'SVM':

        Y_predict, model = SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params)

    elif Parameters['algorithm'] == 'RVM':

        Y_predict, model = RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params)

    elif Parameters['algorithm'] == 'GPR':

        Y_predict, model = GPR(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params)

    return Y_predict, {'model' : model, 'algorithm' : Parameters['algorithm'], 'test_size' : params['test_size']}


# In[9]:

"""

takes in XTest ( which is a zipped form of input and output data tuples ) and a trained model
evaluates the performance of the model

"""
def TestMyClassifier(XTest, EstParameters):
    model = EstParameters['model']
    
    Xactual, Yactual = unzipData(XTest)
    Ypred = []
    
    algorithm = EstParameters['algorithm']
    
    if algorithm == 'GPR' or algorithm ==  'RVM':
        X_actual = reduce_dimensions(X_actual)
        
    for x in X:
        probabilities  = model.predict_proba(X)
        max_probabilty = max(p)
        
        print probabilities
        
        y = p.index(max_probability)
        Ypred.append(y)
        
    return Ypred


# In[10]:

"""

performs K-fold cross validation and selects the best model to prevent overfitting
returns the array of confusion matrices and estimated parameter models for every fold

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
        
        Y_predicted, EstParameter = MyTrainClassifier(En, Vn, {'algorithm':'RVM', 'training_mode':False})
        
        _, Y_validate = unzipData(Vn)
        
        algorithm = EstParameter['algorithm']
        
        if algorithm == 'GPR' or algorithm ==  'RVM':
            Y_validate = Y_validate[:EstParameter['test_size']]
        
        Cn, acc = MyConfusionMatrix(Y_predicted, Y_validate)
        
        EstConfMatrices.append(Cn)
        EstParameters.append(EstParameter)
        
        accuracies.append(acc)
        
        j = j + 1
    
    print ""
    
    best_model_idx = accuracies.index(max(accuracies))
    best_model = EstParameters[best_model_idx]['model']
    
    X, Y = unzipData(XTrain)
    
    algorithm = EstParameters[best_model_idx]['algorithm']
    
    if algorithm == 'GPR' or algorithm ==  'RVM':
        X = reduce_dimensions(X)
        
    YTrain = best_model.predict(X)
    
    print "overall confusion matrix :"
    
    ConfMatrix, acc = MyConfusionMatrix(YTrain, Y)
    
    return YTrain, EstParameters, EstConfMatrices, ConfMatrix


# In[ ]:

data = zipData(inputData, outputData)

MyCrossValidate(data, 5)

