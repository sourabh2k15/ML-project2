
# coding: utf-8

# # ML Project2

# In[66]:

"""

Utility methods

"""

from sklearn import decomposition

# do PCA to reduce dimensionality, advisable for RVM and GPR to reduce training time
def reduce_dimensions(data):
    pca = decomposition.PCA(n_components = 7)

    X = pca.fit_transform(data)
    return X

# reads model from pickled object file
def readObj(name):
    with open(cfg.path + name, 'rb') as input:
        clf = pickle.load(input)

    return clf

# writes model to a pickled object file
def writeObj(name, obj):
    with open(cfg.path + name, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#zips input-output data into list of tuples
def zipData(X, Y):
    return zip(X, Y)

#unzips the data and converts output [1, -1...] to classLabels
def unzipData(X):
    x, y = zip(*X)

    x = np.array(list(x))
    y = np.array([np.where(output == 1)[0][0] for output in list(y)])

    return x,y


# ## Confusion Matrix

# In[76]:

"""

pretty prints confusion matrix and returns confusion matrix and accuracy score

@param   Y                   predicted labels

@param   ClassLabels         actual / true labels

"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd

def MyConfusionMatrix(Y, ClassNames):
    
    conf_matrix = confusion_matrix(Y, ClassNames)
    conf_matrix = np.around(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
    
    accuracy = accuracy_score(Y, ClassNames)

    ClassLabels = range(len(conf_matrix))

    columns = tuple(ClassLabels)
    rows = tuple(ClassLabels)

    df = pd.DataFrame(data=conf_matrix, columns=ClassLabels)

    print "\nconfusion matrix: \n"
    print df

    print "\n"
    print "accuracy: ", accuracy

    return conf_matrix, accuracy


# ```
# 5 is the garbage class
# 
# overall confusion matrix (SVM):
# 
#       0     1     2     3     4     5
# 0  4969    12    27    11    16     0
# 1    14  4884     4    96     0     0
# 2    13    10  4924    13    25     0
# 3     0    93     7  4846    27     0
# 4     4     1    38    34  4932     0
# 5     0     0     0     0     0  5000
# ```

# ## One vs One Trainer

# In[77]:

"""
All Pairs Training

we rolled out our own OneVsOneClassifier which takes in a binary classifier and trains nC2 classifiers
this was done as the default sklearn OneVsOne doesn't support probability estimates for the classes
which would be useful for determining garbage class inputs

"""
class OneVsOne:
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
        elif self.model_ == 'svm':
            return SVC(C=100, kernel='linear', probability=True)

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

                    probabilities[i] += clf.predict_proba([x])[0][0]
                    probabilities[j] += clf.predict_proba([x])[0][1]

            probabilities = [p / 10.0 for p in probabilities]

            Y.append(probabilities.index(max(probabilities)))

        return Y


# 
# example usage of OneVsOne : 
# 
# ```
# clf = OneVsOne('svm')
# clf.fit(X, Y)
# 
# clf.predict(X)
# 
# ```

# ## SVM

# In[78]:

"""
SVM (Support Vector Machine):

performs grid search to compute optimal hyper-parameters
uses those hyper-parameters for the estimator, fits it on the training data

returns trained model and writes it to file for transfer learning

"""

from sklearn.metrics import accuracy_score

def SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params):

    hyper_param_grid = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]

    train = params['train']

    if train:
        estimator = GridSearchCV(SVC(decision_function_shape='ovo', probability=True), hyper_param_grid, cv=3, scoring='precision_macro')

        print "SVM: executing grid search to find optimal hyper-parameters"

        estimator.fit(X_hyper, Y_hyper)
        clf = estimator.best_estimator_
        
        print "found best hyperparameters:"

        print estimator.best_params_
        print "training the estimator"

        clf.fit(X_train, Y_train)
        print("number of support vectors:", len(clf.support_))
        
        writeObj('svm_model.pkl', clf)

        Y_pred = clf.predict(X_validate)
        return Y_pred, clf

    else:
        clf = readObj('svm_model.pkl')
        print("number of support vectors:", len(clf.support_))
        
        Y_pred = clf.predict(X_validate)
        return Y_pred, clf


# ## RVM

# In[79]:

"""
RVM (Relevance Vector Machine):

uses PCA to reduce dimensionality as RVM training takes a long time
also uses a subset of training data to save time


returns trained model and writes it to file for transfer learning

"""
from skrvm import RVC

def RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params):
    clf = RVC(n_iter=100, tol=0.1)
    start = time.clock()

    X_train_reduced = X_train
    X_validate_reduced = X_validate

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


# ## GPR

# In[80]:

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

    X_train_reduced = X_train
    X_validate_reduced = X_validate

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


# ## Training

# In[81]:

import numpy as np
import time

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

def MyTrainClassifier(XEstimate, XValidate, Parameters):

    X_train, Y_train = unzipData(XEstimate)

    # sampling a small amount of training data for finding optimal hyper-parameters
    X_hyper = X_train[:cfg.hyper_train_size, :]
    Y_hyper = Y_train[:cfg.hyper_train_size]

    X_validate, Y_validate = unzipData(XValidate)

    train = Parameters['training_mode']
    params = { 'train' : train, 'train_size' : cfg.train_size, 'test_size' : cfg.test_size }

    if Parameters['algorithm'] == 'SVM':

        Y_predict, model = SVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params)

    elif Parameters['algorithm'] == 'RVM':

        Y_predict, model = RVM(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params)

    elif Parameters['algorithm'] == 'GPR':

        Y_predict, model = GPR(X_hyper, Y_hyper, X_train, Y_train, X_validate, Y_validate, params)

    return Y_predict, {'model' : model, 'algorithm' : Parameters['algorithm'], 'test_size' : params['test_size']}


# # Testing

# In[82]:

"""

takes in XTest ( which is a zipped form of input and output data tuples ) and a trained model.
evaluates the performance of the model

"""
def TestMyClassifier(XTest, EstParameters):
    model = EstParameters['model']

    Xactual, _ = unzipData(XTest)
    Ytest = model.predict(Xactual)

    return Ytest


# ## K-Fold Cross-Validation

# In[83]:

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

        Y_predicted, EstParameter = MyTrainClassifier(En, Vn, {'algorithm' : cfg.algorithm, 'training_mode' : cfg.train})

        _, Y_validate = unzipData(Vn)

        algorithm = EstParameter['algorithm']

        if algorithm == 'GPR' or algorithm ==  'RVM':
            Y_validate = Y_validate[:EstParameter['test_size']]

        Cn, acc = MyConfusionMatrix(Y_predicted, Y_validate)

        EstParameter['accuracy'] = acc

        EstConfMatrices.append(Cn)
        EstParameters.append(EstParameter)

        accuracies.append(acc)

        j = j + 1

    print ""

    best_model_idx = accuracies.index(max(accuracies))
    best_model = EstParameters[best_model_idx]['model']

    X, Y = unzipData(XTrain)

    algorithm = EstParameters[best_model_idx]['algorithm']
    YTrain = best_model.predict(X)

    print "overall confusion matrix :"

    ConfMatrix, acc = MyConfusionMatrix(YTrain, Y)

    return YTrain, EstParameters, EstConfMatrices, ConfMatrix, best_model_idx


# ## Demo Usage

# In[84]:

"""
driver program

"""

from scipy.io import loadmat
import config as cfg

print cfg.algorithm

#loading input - output data and explore the dataset
inputDataPath  = 'data/Proj2FeatVecsSet1.mat'
outputDataPath = 'data/Proj2TargetOutputsSet1.mat'

inputDataObj  = loadmat(inputDataPath)
outputDataObj = loadmat(outputDataPath)

inputData  = np.array(inputDataObj['Proj2FeatVecsSet1'])
outputData = np.array(outputDataObj['Proj2TargetOutputsSet1'])

# adding extra class to outputs
outputData = np.array([np.append(output, [-1]) for output in outputData])

# mixing in 5000 samples of noise
X_garbage = np.reshape(np.random.rand(300000), (5000, 60))
Y_garbage = np.array([np.array([-1,-1,-1,-1,-1, 1])]*5000)

X = np.vstack((inputData, X_garbage))
Y = np.vstack((outputData, Y_garbage))

# noise added training data packed
data = zipData(X, Y)

# without noise training uncomment
# data = zipData(inputData, outputData)

# 5-fold cross-validation to obtain best model that prevents over-fitting
Y_pred, EstParams, EstConfMatrices, ConfMatrix, best_idx = MyCrossValidate(data, cfg.k)

# testing garbage class functionality
garbage = [[5]*60]
garbage2 = np.random.rand(1,60)

Y_test = TestMyClassifier(zipData([X[0]], [1]), EstParams[best_idx])
Y_test = TestMyClassifier(zipData([X[50]], [1]), EstParams[best_idx])

print "garbage test1, predicted label : ", TestMyClassifier(zipData(garbage, [1]), EstParams[best_idx])
print "garbage test2, predicted label: ", TestMyClassifier(zipData(garbage2, [1]), EstParams[best_idx])


# In[ ]:



