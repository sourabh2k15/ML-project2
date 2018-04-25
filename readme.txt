# Prereqs Install : 
----------------------

1)jupyter, scipy, sklearn, numpy, pandas:

	pip install numpy scipy ipython jupyter pandas

2) skrvm : 

	pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip

# Run Instructions : 
----------------------

1) cd into the project folder in your terminal
2) type in `jupyter notebook` which will open a browser session
3) edit config.ini to set the algorithm you want to run and other options available

# Folder Structure :
----------------------
- data/
- transfer_learning/
- result/
- project2.py
- project2.ipynb
- config.py

* data/                             contains the input output data mat files
* transfer_learning/                has pretrained model objects saved as pickle files
* results/                           has logs of previous runs for all algorithms
* project2.py                       the .py file for the jupyter notebook, provided to run for faster training
* project2.ipynb                    jupyter notebook that contains all code
* config.py                         configuration file from which parameters are read     


# Configuration Options (config.py)
----------------------

1) algorithm           : could be one of SVM | RVM | GPR
2) train               : True | False , if false it loads one of the pretrained models
3) train_size          : for GPR and RVM to reduce training time , we can train on a reduced train set
4) hyper_train_size    : the amount of training data used to run GridSearchCV to find optimal hyper-parameters


# Error Notes 
-----------------------

the library skrvm uses python's process_pool to parallelize the jobs in training, this could crash at times as they go out of sync. This is an in built library error, simply restart the training, there is no bug in our code

# Notes / Experiments / Learning
---------------------------------
* classes 0-4 are the normal classes, class 5 is the garbage class
* various experiments were carried out, which have been mentioned below

1) to reduce the training time for RVM and GPR we did PCA to reduce dimensionality of input from 60 -> 7, which gave acceptable results without much loss in accuracy

2) in order to detect the garbage class, we based our predictions based on probability estimates, and we tried using both OneVsOne and OneVsRest but the probability estimate even for the garbage input is pretty high for one of the actual classes

3) so we mixed some noise in our traininf data itself and added an extra class to the training output classes, this led to bad performance in GPR and RVM as PCA components were now less varied

4) we got rid of PCA, did OneVsRest with noise mixed in training data for garbage class detection and got a good result on validation with entire dataset

5) in order to reduce training time for GPR and RVM we simply used a smaller subset of training data which would still give pretty good results on validation with entire dataset

6) SVM is very fast!! with a very good performance but RVM outperforms it. In fact RVM and GPR need relatively less data in order to give comparable accuracy to SVM

7) RVM vs GPR : RVM is very fast when it comes to predictions, and it classifies data better than SVM. GPR is the slowest to train among all 3 algorithms

8) OneVsOne training / all-pairs is more accurate than OneVsRest training but takes more time as it needs to train NC2 classifiers as compared to one-vs-all which would only need classifiers linear in the number of classes.

9) The number of relevance vectors used in RVM is linear in the size of training data unlike SVM
