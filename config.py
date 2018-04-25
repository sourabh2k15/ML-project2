# the classifier you want to run : SVM | RVM | GPR
algorithm = 'RVM'

# if train is False, it will load the saved model from a previous run and run validation
train = True

# k-folds 
k = 5

#size of dataset used for finding optimal hyper-parameters
hyper_train_size = 500

# used only for RVM , GPR as training takes a long time, so we use a reduced set
# note : after k-fold, the model is run for validation on the entire dataset to obtain true performance measures

test_size = 200
train_size = 500

# filepath for saved models
path = 'transfer_learning/'

