#!/usr/bin/env python
# coding: utf-8

# # Project Phase 3 - Gender & Speaker Recognition

# Importing libraries

# In[1]:


import os
import sys
import python_speech_features as mfcc
from scipy.io.wavfile import read
import numpy as np
import glob


# Functions to get feature vectors

# In[2]:


def get_MFCC(audio, sr):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = True)
    return np.mean(features, axis=0)


# In[3]:


def get_features(data):
    x, m = np.shape(data)
    In = []
    Out = []
    for i in range (x):
        In = data[i]
        feature = get_MFCC(In[1], In[0])
        Out.append(feature) 
    return Out


# # Gender Data

# Made 4 np arrays:
# - gtrain_X & gtrain_Y
# - gtest_X & gtest_Y
# 
# X data sets have 13 features in reach row for each input .wav file
# 
# Y data has corresponding gender (2 classes - 0 and 1)

# In[4]:


suffix = "_F";
train = []
gtrain_Y = []
train_list = glob.glob(os.path.join(os.getcwd(), "Gender_Recognition/Train"))
for train_path in train_list:
    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            filepath = subdir + os.sep + file
            if os.path.dirname(filepath).endswith(suffix):
                temp = 1;
            else:
                temp = 0;
            if filepath.endswith(".wav"):
                with open(filepath) as train_input:
                    train.append(read(filepath))
                    gtrain_Y.append(temp)
gtrain_Y = list(map(int, gtrain_Y)) 


# In[5]:


test = []
gtest_Y = []
test_list = glob.glob(os.path.join(os.getcwd(), "Gender_Recognition/Test"))
for test_path in test_list:
    for subdir, dirs, files in os.walk(test_path):
        for file in files:
            filepath = subdir + os.sep + file
            if os.path.dirname(filepath).endswith(suffix):
                temp = 1;
            else:
                temp = 0;
            if filepath.endswith(".wav"):
                with open(filepath) as test_input:
                    test.append(read(filepath))
                    gtest_Y.append(temp)
gtest_Y = list(map(int, gtest_Y)) 


# In[6]:


gtrain_X = get_features(train)
gtest_X = get_features(test)


# # Speaker Data

# Made 4 np arrays:
# - strain_X & strain_Y
# - stest_X & stest_Y
# 
# X data sets have 13 features in reach row for each input .wav file
# 
# Y data has corresponding speaker (142 classes - 0 to 141)

# In[7]:


train = []
trainY = []
train_list = glob.glob(os.path.join(os.getcwd(), "Speaker_Recognition/Train"))
for train_path in train_list:
    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            filepath = subdir + os.sep + file
            direc = os.path.dirname(filepath)
            direc = direc[-5:]
            temp = direc[:-2]
            if filepath.endswith(".wav"):
                with open(filepath) as train_input:
                    train.append(read(filepath))
                    trainY.append(temp)
trainY = list(map(int, trainY)) 
strain_Y = np.array(trainY) - 1


# In[8]:


test = []
testY = []
test_list = glob.glob(os.path.join(os.getcwd(), "Speaker_Recognition/Test"))
for test_path in test_list:
    for subdir, dirs, files in os.walk(test_path):
        for file in files:
            filepath = subdir + os.sep + file
            direc = os.path.dirname(filepath)
            direc = direc[-5:]
            temp = direc[:-2]
            if filepath.endswith(".wav"):
                with open(filepath) as test_input:
                    test.append(read(filepath))
                    testY.append(temp)
testY = list(map(int, testY)) 
stest_Y = np.array(testY) - 1


# In[9]:


strain_X = get_features(train)
stest_X = get_features(test)


# Starting sklearn implementation:

# In[10]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[11]:


def evaluation(test_Y, result):
    print('Accuracy is: ', accuracy_score(result, test_Y)*100,'%')
    print('Confusion Matrix is:') 
    print(confusion_matrix(result, test_Y))
    print(classification_report(test_Y, result))
    return None


# # Part 1

# In[12]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
mlp = MLPClassifier(max_iter = 5000, activation = 'logistic', solver = 'sgd', random_state = 1)


# In[13]:


paramgrid = {'hidden_layer_sizes': [(128,64), (64), (64,32), (32)],
             'learning_rate_init': [0.4, 0.1, 0.01]}
gcs = GridSearchCV(mlp, paramgrid, scoring = 'f1_macro', cv = 3)


# Gender Recognition:

# In[14]:


grid_res = gcs.fit(gtrain_X, gtrain_Y)
best_params_g = grid_res.best_params_


# In[15]:


print('The best parameters are:')
print(best_params_g)


# In[16]:


result = grid_res.predict(gtest_X)
evaluation(gtest_Y, result)


# Speaker Recognition:

# In[17]:


grid_res_s = gcs.fit(strain_X, strain_Y)
best_params_s = grid_res_s.best_params_


# In[18]:


print('The best parameters are:')
print(best_params_s)


# In[19]:


result = grid_res_s.predict(stest_X)
evaluation(stest_Y, result)


# # Part 2

# In[20]:


from sklearn.svm import LinearSVC
svc = LinearSVC(max_iter = 1e5)


# Gender Recognition:

# In[21]:


gend = svc.fit(gtrain_X, gtrain_Y)
result = gend.predict(gtest_X)
evaluation(gtest_Y, result)


# Speaker Recognition:

# In[22]:


speak = svc.fit(strain_X, strain_Y)
result = speak.predict(stest_X)
evaluation(stest_Y, result)


# # Part 3

# In[23]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()


# Gender Recognition:

# In[24]:


GNB.fit(gtrain_X, gtrain_Y)
result = GNB.predict(gtest_X)
evaluation(gtest_Y, result)


# Speaker Recognition:

# In[25]:


GNB.fit(strain_X, strain_Y)
result = GNB.predict(stest_X)
evaluation(stest_Y, result)


# In[ ]:




