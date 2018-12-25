# -*- coding: utf-8 -*-
# using python3.6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from csv import reader

##
# @dev: load data as array
# @params: filename -> data file
def load_data(filename):
    data = pd.read_table(filename,header=None,sep=',')
    dataset = np.array(data)
    return dataset

##
# @dev: use trained model to generate outcomes
# @params: clf -> model trained
# @params: X -> samples of dataset
# @params: y -> target of dataset
# @params: n_folds -> number of folds
def run_kfold(clf, X, y, n_folds):
    kf = KFold(n_folds)
    outcomes = []
    fold = 0
    
    for train_index, test_index in kf.split(X):
        fold = fold + 1
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        
        clf.fit(X_train,y_train)
        pre = clf.predict(X_test)
        prob_pre = clf.predict_proba(X_test) #calculate possibility ourcome，sum of all possibility of a label is 1
        pre_val = prob_pre[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pre_val)
        roc_auc = auc(fpr, tpr)
        outcomes.append(roc_auc)

    mean_outcome = np.mean(outcomes)
    
    return mean_outcome

##
# @dev: test the model and plot the outcome
def test_d_m():
    # load data
    path_f = '../data/Test1_features.dat'
    path_l = '../data/Test1_labels.dat'

    X, y = (load_data(path_f), load_data(path_l).ravel())
    min_samples_leaf = range(40,60,1)
    max_features=np.linspace(0.1,1)
    scores = []
    i=0
    # training model and validation test
    for min_samples in min_samples_leaf:
        for max_feature in max_features:
            regr = ensemble.RandomForestClassifier(max_depth=3,max_features=max_feature,min_samples_leaf=min_samples) 
            scores.append(run_kfold(regr,X,y, 3))
            i=i+1
            print('step:',i)
            print(scores)

    ## plot 3-D table
    min_samples_leaf,max_features = np.meshgrid(min_samples_leaf,max_features)
    scores = np.array(scores).reshape(min_samples_leaf.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import  MultipleLocator
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(min_samples_leaf,max_features,scores,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    xmajorLocator = MultipleLocator(1)
    ymajorLocator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel("param1:min_sampls_leaf")
    ax.set_ylabel("param2:max_features")
    ax.set_zlabel("score")
    ax.set_title("task1：3-fold AUC score")
    plt.show()
    
test_d_m()