#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:50:31 2020

@author: vinoothna
"""

import numpy  as np # Import numpy
import matplotlib.pyplot as plt #import matplotlib library

from matplotlib import interactive
interactive(True)
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats 
import sklearn
from sklearn.preprocessing import normalize
from scipy import signal
from scipy import stats
from numpy import linalg as LA
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.interpolate import CubicSpline
from scipy.fftpack import fft, ifft
import scipy as sc      
from scipy import stats 
from scipy.fftpack import fft
from scipy.signal import spectrogram as sp
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
#from sklearn.metrics import plot_confusion_matrix
fname ="train_set.csv"
#all_data = [line.rstrip('[').rstrip(']') for line in open(fname)];
all_data=[[float(num) for num in line.rstrip('\n').replace('[',' ').replace(']',' ').replace(',',' ').split()] for line in open(fname)];

fname1 ="test_set.csv"
##all_data = [line.rstrip('[').rstrip(']') for line in open(fname)];
all_data1=[[float(num) for num in line.rstrip('\n').replace('[',' ').replace(']',' ').replace(',',' ').split()] for line in open(fname1)];
#
all_data1= np.asarray(all_data1) 
train=np.empty((50,))
test=np.empty((50,))
#
train_y=np.empty((1,))
test_y=np.empty((1,))
u=0
for i in range(5000):
    
       
    x = all_data[i][0:50]
    #new_im.show()
    x=np.asarray(x)
    y= all_data[i][54]
    y=np.asarray(y)
    
    
    #x=(x-np.min(x))/(np.max(x)-np.min(x))
    train_y=np.vstack((train_y,y))
    train = np.vstack((train, x))










    
train=train[1:][:] 
train_y=train_y[1:][:].reshape((-1,1)) 
for i in range(1000):
    
       
    x = all_data1[i][0:50]
    #new_im.show()
    x=np.asarray(x)
    y= all_data1[i][54]
    y=np.asarray(y)
    
    
  #  x=(x-np.min(x))/(np.max(x)-np.min(x))
    test_y=np.vstack((test_y,y))
    test = np.vstack((test, x))    
    
    
test=test[1:][:] 
test_y=test_y[1:][:].reshape((-1,1)) 
 
#
#
#
#
#
import math


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000)
# fit model
model.fit(train, train_y)
predictions = model.predict(test)
mse = sklearn.metrics.mean_squared_error(test_y, predictions)

rmse = math.sqrt(mse)

print(rmse)


#score = cross_val_score(model,train, train_y, scoring='neg_root_mean_squared_error')
# summarize performance
#n_scores = absolute(score)
#print('Result: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

